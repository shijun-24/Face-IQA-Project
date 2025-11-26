# Face Image Quality Assessment (Face-IQA)

This repository contains the official implementation of a Face Image Quality Assessment model based on **ConvNeXt-Base**. The model utilizes a consistency learning strategy and is validated using **5-Fold Cross-Validation**.

##  Features
- **Backbone**: ConvNeXt-Base (Pretrained on ImageNet)
- **Preprocessing**: MTCNN for automatic face detection and cropping.
- **Loss Function**: Combination of MSE, PLCC Loss, and Rank Loss.
- **Metrics**: SRCC, PLCC, KRCC, RMSE.
- **Ensemble**: 5-Fold ensemble inference for robust scoring.

##  Dataset Structure
Please organize your dataset as follows:
```text
data/
└── Annoted_Dataset/
    ├── RAW/             # Original raw images
    ├── All/             # Reference/All images
    └── BT-Scores.xlsx   # Ground truth labels
 Installation
Bash

# Clone the repository
git clone [https://github.com/YourUsername/Face-IQA-Project.git](https://github.com/YourUsername/Face-IQA-Project.git)
cd Face-IQA-Project

# Install dependencies
pip install -r requirements.txt
 Usage
1. Training
To train the model with 5-fold cross-validation:

Bash

python train.py
Note: The script will automatically perform MTCNN preprocessing if not already done.

2. Inference
To evaluate image quality using the trained ensemble models:

Bash

python inference_auto.py
