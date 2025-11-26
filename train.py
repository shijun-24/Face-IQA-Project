import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import KFold
from facenet_pytorch import MTCNN
from collections import defaultdict
import gc  # 引入垃圾回收

# 1. 配置

DATA_DIR = "/root/autodl-tmp/prj/data/Annoted_Dataset"
RAW_IMAGES_DIR = os.path.join(DATA_DIR, "RAW")
ALL_IMAGES_DIR = os.path.join(DATA_DIR, "All")
RAW_PROCESSED_DIR = os.path.join(DATA_DIR, "RAW_processed_convnext")
SCORES_FILE = os.path.join(DATA_DIR, "BT-Scores.xlsx")
OUTPUT_DIR = "./outputs_convnext_5fold"  # 修改输出目录

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_PROCESSED_DIR, exist_ok=True)

IMAGE_SIZE = (384, 384)
BATCH_SIZE = 32
NUM_WORKERS = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
SEED = 2025
DEVICE = 'cuda'
USE_MTCNN_PROCESSED = True
PATIENCE = 8  # 早停容忍次数：8个Epoch不提升则停止
K_FOLDS = 5  # 折数

print(f"| Device: {torch.cuda.get_device_name(0)}")


# 2. 预处理 (保持不变)
def preprocess_raw_folder():
    print("\n" + "=" * 40)
    print("   [Step 1] MTCNN 抠图")
    print("=" * 40)
    if len(os.listdir(RAW_PROCESSED_DIR)) > 100:
        print(f"检测到预处理文件，跳过...")
        return
    mtcnn = MTCNN(image_size=256, margin=0, min_face_size=40,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                  select_largest=True, device=DEVICE)
    image_files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in tqdm(image_files, desc="Processing"):
        src = os.path.join(RAW_IMAGES_DIR, filename)
        dst = os.path.join(RAW_PROCESSED_DIR, os.path.splitext(filename)[0] + ".png")
        try:
            img = Image.open(src).convert('RGB')
            mtcnn(img, save_path=dst)
        except:
            img = img.resize((256, 256))
            img.save(dst)
    print("预处理完成！")


# 3. 模型 (保持不变)
class ConsistencyConvNeXt(nn.Module):
    def __init__(self):
        super(ConsistencyConvNeXt, self).__init__()
        # print("正在加载 ConvNeXt-Base 权重...") # 注释掉以减少日志刷屏
        self.backbone = models.convnext_base(weights='DEFAULT')
        num_ftrs = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        self.score_head = nn.Sequential(
            nn.Linear(num_ftrs * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, raw_imgs, all_imgs):
        f_raw = self.backbone(raw_imgs)
        f_all = self.backbone(all_imgs)
        f_raw = torch.flatten(f_raw, 1)
        f_all = torch.flatten(f_all, 1)
        combined = torch.cat((f_raw, f_all), dim=1)
        score = self.score_head(combined)
        return score.squeeze()


# 4. 损失函数 (保持不变)
class AdvancedLoss(nn.Module):
    def __init__(self, plcc_w=1.0, rank_w=1.0):
        super(AdvancedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.plcc_w = plcc_w
        self.rank_w = rank_w
        self.relu = nn.ReLU()

    def forward(self, y_pred, y_true):
        loss_mse = self.mse(y_pred, y_true)
        y_pred_mean, y_true_mean = torch.mean(y_pred), torch.mean(y_true)
        y_pred_centered, y_true_centered = y_pred - y_pred_mean, y_true - y_true_mean
        eps = 1e-8
        cov = torch.sum(y_pred_centered * y_true_centered)
        pred_std = torch.sqrt(torch.sum(y_pred_centered ** 2) + eps)
        true_std = torch.sqrt(torch.sum(y_true_centered ** 2) + eps)
        loss_plcc = 1.0 - (cov / (pred_std * true_std))
        n = y_pred.size(0)
        if n > 1:
            idx = torch.randperm(n)
            y_pred_s, y_true_s = y_pred[idx], y_true[idx]
            diff_true = y_true - y_true_s
            diff_pred = y_pred - y_pred_s
            loss_rank = torch.mean(self.relu(-diff_true * diff_pred))
        else:
            loss_rank = torch.tensor(0.0).to(y_pred.device)
        return loss_mse + self.plcc_w * loss_plcc + self.rank_w * loss_rank


# 5. 数据准备 (修改为支持外部传入 fold)
class FaceDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        raw_path, all_path, score = self.data_list[idx]
        try:
            raw_img = Image.open(raw_path).convert('RGB')
            all_img = Image.open(all_path).convert('RGB')
            if self.transform:
                raw_img = self.transform(raw_img)
                all_img = self.transform(all_img)
            return raw_img, all_img, torch.tensor(score, dtype=torch.float32)
        except:
            return torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1])), torch.zeros(
                (3, IMAGE_SIZE[0], IMAGE_SIZE[1])), torch.tensor(-1.0)


# 新增：一次性读取所有数据并准备划分
def prepare_all_data():
    print(f"\n[Step 2] 读取数据源并生成划分索引...")
    if USE_MTCNN_PROCESSED and os.path.exists(RAW_PROCESSED_DIR) and len(os.listdir(RAW_PROCESSED_DIR)) > 0:
        raw_source = RAW_PROCESSED_DIR
        print(f"  - 使用高清源(MTCNN): {raw_source}")
    else:
        raw_source = RAW_IMAGES_DIR
        print(f"  - 使用原始源: {raw_source}")

    if not os.path.exists(ALL_IMAGES_DIR): raise FileNotFoundError(f"找不到 All 文件夹")

    raw_map = {os.path.splitext(f)[0]: os.path.join(raw_source, f) for f in os.listdir(raw_source) if
               f.lower().endswith(('.png', '.jpg'))}
    all_map = {os.path.splitext(f)[0]: os.path.join(ALL_IMAGES_DIR, f) for f in os.listdir(ALL_IMAGES_DIR) if
               f.lower().endswith(('.png', '.jpg'))}

    df = pd.read_excel(SCORES_FILE)
    id_to_samples = defaultdict(list)
    valid_count = 0
    for _, row in df.iterrows():
        try:
            all_id = str(int(row[df.columns[0]]))
            score = float(row['人脸一致性维度分数'])
            raw_id = str(((int(all_id) - 1) // 8) + 1)
            if raw_id in raw_map and all_id in all_map:
                sample = (raw_map[raw_id], all_map[all_id], score)
                id_to_samples[raw_id].append(sample)
                valid_count += 1
        except:
            continue

    unique_ids = sorted(list(id_to_samples.keys()))
    random.seed(SEED)
    random.shuffle(unique_ids)

    print(f"  - 有效 ID 数: {len(unique_ids)}")
    print(f"  - 总样本对: {valid_count}")

    return unique_ids, id_to_samples


# 新增：根据ID列表创建 DataLoader
def get_fold_dataloaders(train_ids, val_ids, id_to_samples):
    train_samples = []
    for uid in train_ids: train_samples.extend(id_to_samples[uid])
    val_samples = []
    for uid in val_ids: val_samples.extend(id_to_samples[uid])

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_loader = DataLoader(FaceDataset(train_samples, train_tf), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(FaceDataset(val_samples, val_tf), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader


# 6. 指标计算 (保持不变)
def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    plcc = pearsonr(y_true, y_pred)[0]
    srcc = spearmanr(y_true, y_pred)[0]
    krcc = kendalltau(y_true, y_pred)[0]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return srcc, plcc, krcc, rmse


# 7. 早停类 (新增)
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = -999.0  # 假设 SRCC+PLCC 越大越好
        self.early_stop = False

    def __call__(self, current_score, model, path):
        # 这里的 score 是 SRCC + PLCC 的和
        if current_score > self.best_score + self.delta:
            self.best_score = current_score
            self.counter = 0
            torch.save(model.state_dict(), path)
            return True  # 保存了新模型
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# 8. 主训练流程 (修改为 5折循环)

# 8. 主训练流程 (修改：包含 KRCC 的 4 参数完整汇总)

def run_training():
    torch.manual_seed(SEED)

    # 获取所有数据和ID映射
    unique_ids, id_to_samples = prepare_all_data()

    # 创建 KFold
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # 记录每一折的最佳结果
    fold_results = []

    print(f"\n" + "=" * 60)
    print(f" 开始 {K_FOLDS} 折交叉验证 (Patience={PATIENCE}) | 指标: SRCC, PLCC, KRCC, RMSE")
    print("=" * 60)

    for fold_idx, (train_ids_idx, val_ids_idx) in enumerate(kf.split(unique_ids)):
        print(f"\n>>> 正在进行第 {fold_idx + 1} / {K_FOLDS} 折训练")

        train_ids = [unique_ids[i] for i in train_ids_idx]
        val_ids = [unique_ids[i] for i in val_ids_idx]

        train_loader, val_loader = get_fold_dataloaders(train_ids, val_ids, id_to_samples)

        # 每一折必须重新初始化
        model = ConsistencyConvNeXt().to(DEVICE)
        criterion = AdvancedLoss(plcc_w=1.5, rank_w=0.5).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        # 初始化早停
        save_path = os.path.join(OUTPUT_DIR, f"best_fold_{fold_idx + 1}.pth")
        early_stopper = EarlyStopping(patience=PATIENCE)

        # 初始化本折最佳记录
        best_fold_metrics = {'SRCC': 0, 'PLCC': 0, 'KRCC': 0, 'RMSE': 0}

        for epoch in range(NUM_EPOCHS):
            # --- Train ---
            model.train()
            train_preds, train_targets = [], []
            loop_desc = f"Fold {fold_idx + 1} Ep {epoch + 1}"

            for raw, all_img, score in tqdm(train_loader, leave=False, desc=loop_desc):
                if score[0] == -1: continue
                raw, all_img, score = raw.to(DEVICE), all_img.to(DEVICE), score.to(DEVICE)

                optimizer.zero_grad()
                with autocast():
                    pred = model(raw, all_img)
                    loss = criterion(pred, score)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_preds.extend(pred.detach().float().cpu().tolist())
                train_targets.extend(score.cpu().tolist())

            scheduler.step()

            # --- Val ---
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for raw, all_img, score in tqdm(val_loader, leave=False, desc="Validating"):
                    if score[0] == -1: continue
                    raw, all_img = raw.to(DEVICE), all_img.to(DEVICE)
                    with autocast():
                        out1 = model(raw, all_img)
                        out2 = model(torch.flip(raw, [3]), torch.flip(all_img, [3]))  # TTA
                        final = (out1 + out2) / 2.0
                    val_preds.extend(final.cpu().tolist())
                    val_targets.extend(score.tolist())

            # --- Metrics ---
            t_srcc, t_plcc, t_krcc, t_rmse = compute_metrics(train_targets, train_preds)
            v_srcc, v_plcc, v_krcc, v_rmse = compute_metrics(val_targets, val_preds)

            current_score = v_srcc + v_plcc

            # 打印日志 (包含 KRCC)
            print(f"[{fold_idx + 1}/{K_FOLDS}][Ep {epoch + 1:02d}] "
                  f"Val SRCC:{v_srcc:.4f} PLCC:{v_plcc:.4f} KRCC:{v_krcc:.4f} RMSE:{v_rmse:.4f}", end="")

            # --- Early Stopping Check ---
            saved = early_stopper(current_score, model, save_path)

            if saved:
                print(" ->  Saved")
                # 更新所有4个指标
                best_fold_metrics = {
                    'SRCC': v_srcc,
                    'PLCC': v_plcc,
                    'KRCC': v_krcc,
                    'RMSE': v_rmse
                }
            else:
                print(f" | Pat {early_stopper.counter}/{PATIENCE}")

            if early_stopper.early_stop:
                print(f"    ! 早停触发")
                break

        print(f"    本折最佳: SRCC={best_fold_metrics['SRCC']:.4f}, KRCC={best_fold_metrics['KRCC']:.4f}")
        fold_results.append(best_fold_metrics)

        # 清理
        del model, optimizer, scaler, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # --- 最终总结 ---
    print("\n" + "=" * 60)
    print(" 5-Fold Cross Validation Results Summary (4 Metrics)")
    print("=" * 60)

    avg_srcc = np.mean([r['SRCC'] for r in fold_results])
    avg_plcc = np.mean([r['PLCC'] for r in fold_results])
    avg_krcc = np.mean([r['KRCC'] for r in fold_results])  # 新增
    avg_rmse = np.mean([r['RMSE'] for r in fold_results])

    for i, res in enumerate(fold_results):
        print(
            f"Fold {i + 1}: SRCC={res['SRCC']:.4f} | PLCC={res['PLCC']:.4f} | KRCC={res['KRCC']:.4f} | RMSE={res['RMSE']:.4f}")

    print("-" * 50)
    print(f"Average: SRCC={avg_srcc:.4f} | PLCC={avg_plcc:.4f} | KRCC={avg_krcc:.4f} | RMSE={avg_rmse:.4f}")
    print("=" * 60)

def main():
    while True:
        print("\n" + "=" * 40)
        print(" ConvNeXt (5-Fold + EarlyStopping)")
        print("=" * 40)
        print("1. [预处理] 运行 MTCNN")
        print("2. [训练]   开始5折交叉验证训练")
        print("0. [退出]")
        c = input("选择: ")
        if c == '1':
            preprocess_raw_folder()
        elif c == '2':
            run_training()
        elif c == '0':
            break


if __name__ == "__main__":
    main()
