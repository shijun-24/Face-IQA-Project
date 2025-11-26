import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from facenet_pytorch import MTCNN
import os
import sys

# ================= 配置区域 =================
# 显卡设置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型保存的文件夹路径
MODEL_DIR = "./outputs_convnext_5fold"

# 图片尺寸 (必须和训练时一致)
IMAGE_SIZE = (384, 384)


# ================= 1. 定义模型结构 (必须与训练一致) =================
class ConsistencyConvNeXt(nn.Module):
    def __init__(self):
        super(ConsistencyConvNeXt, self).__init__()
        # 推理时不需要下载预训练权重，因为我们会加载我们自己训练好的
        self.backbone = models.convnext_base(weights=None)
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


# ================= 2. 初始化工具 =================

# 初始化 MTCNN (用于自动抠图)
print("正在初始化 MTCNN 人脸检测器...")
mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE)

# 定义预处理 (标准化) - 必须和训练代码的 val_tf 完全一致
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
val_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])


# ================= 3. 加载 5 折模型 =================
def load_ensemble_models():
    models_list = []
    print(f"正在加载 5 折模型权重...")
    for i in range(1, 6):
        path = os.path.join(MODEL_DIR, f"best_fold_{i}.pth")
        if not os.path.exists(path):
            print(f"  [跳过] 找不到 {path}")
            continue

        try:
            model = ConsistencyConvNeXt().to(DEVICE)
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()  # 开启评估模式
            models_list.append(model)
            print(f"  [成功] 模型 {i} 已加载")
        except Exception as e:
            print(f"  [错误] 模型 {i} 加载失败: {e}")

    if not models_list:
        print("错误: 没有加载到任何模型！请检查 MODEL_DIR 路径。")
        sys.exit(1)

    return models_list


# ================= 4. 智能预测函数 (核心) =================
def predict_one_pair(models_list, raw_path, all_path):
    """
    输入:
        raw_path: 原始图片的路径 (可以是未裁剪的大图)
        all_path: 全景/参考图路径
    输出:
        预测分数 (float)
    """
    try:
        # --- 读取图片 ---
        raw_img_origin = Image.open(raw_path).convert('RGB')
        all_img = Image.open(all_path).convert('RGB')
    except Exception as e:
        print(f"图片读取失败: {e}")
        return None

    # --- 关键步骤: 自动人脸裁剪 (Auto-Crop) ---
    # 我们使用 mtcnn.detect 获取人脸框，然后自己 crop，
    # 这样可以确保后续 transform 流程和训练时完全一致。
    boxes, _ = mtcnn.detect(raw_img_origin)

    if boxes is not None:
        # 找到人脸了，取第一个(概率最大/面积最大)
        box = boxes[0]
        # 转换为整数坐标
        box = [int(b) for b in box]
        # 使用 PIL 进行裁剪
        raw_img_crop = raw_img_origin.crop(box)
        # print(f"  > 检测到人脸，已自动裁剪。")
    else:
        # 没找到人脸，回退方案：直接使用原图
        print(f"  > 警告: 未检测到人脸，将使用原图预测 (可能会不准)。")
        raw_img_crop = raw_img_origin

    # --- 预处理转 Tensor ---
    raw_tensor = val_tf(raw_img_crop).unsqueeze(0).to(DEVICE)
    all_tensor = val_tf(all_img).unsqueeze(0).to(DEVICE)

    # --- 5模型集成预测 ---
    total_score = 0.0
    with torch.no_grad():
        for model in models_list:
            # 这里的 raw_tensor 已经是裁剪好的人脸了，模型看着很亲切
            out1 = model(raw_tensor, all_tensor)

            # TTA (水平翻转测试) - 增加鲁棒性
            out2 = model(torch.flip(raw_tensor, [3]), torch.flip(all_tensor, [3]))

            fold_score = (out1 + out2) / 2.0
            total_score += fold_score.item()

    avg_score = total_score / len(models_list)
    return avg_score


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载模型 (只需加载一次)
    ensemble_models = load_ensemble_models()

    print("\n" + "=" * 40)
    print("  开始预测 (支持自动人脸裁剪)")
    print("=" * 40)

    # 2. 指定你要测试的图片路径 (哪怕是 RAW 原图也可以！)
    # 请修改这里为你真实的路径
    my_test_raw = "/root/autodl-tmp/prj/data/Annoted_Dataset/RAW/1.png"
    my_test_all = "/root/autodl-tmp/prj/data/Annoted_Dataset/All/1.jpg"

    if os.path.exists(my_test_raw) and os.path.exists(my_test_all):
        print(f"正在处理:\n  RAW: {my_test_raw}\n  ALL: {my_test_all}")

        score = predict_one_pair(ensemble_models, my_test_raw, my_test_all)

        if score is not None:
            print("-" * 30)
            print(f" >> 最终预测得分: {score:.4f}")
            print("-" * 30)
    else:
        print(f"错误: 找不到测试图片文件，请检查路径。")
