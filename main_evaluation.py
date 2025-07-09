import os
import numpy as np
from data_loader import testing_data_loader
from metrics import confusion_matrix
from metrics.hausdorff_distance import evaluate_hd_metrics

# === 使用者可調參數 ===
IMG_HEIGHT = 512
IMG_WIDTH = 512

IMAGE_DIR = 'test_oct_image/'
GT_MASK_DIR = 'test_mask_ground_truth/'
PRED_MASK_DIR = 'pred_mask_testing/'

USE_GPU = True 


X_test, Y_test, Z_test = testing_data_loader.load_test_data(
    image_dir=IMAGE_DIR,
    gt_mask_dir=GT_MASK_DIR,
    pred_mask_dir=PRED_MASK_DIR,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
)

accuracy, IoU, precision, recall, dsc = confusion_matrix.confusion_matrix(Y_test, Z_test)

hd, hd95, assd = evaluate_hd_metrics(Y_test, Z_test, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, use_gpu=USE_GPU)

# === Output analysis results ===
print("\n=== Analysis Results ===")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Dice : {dsc:.4f}")
print(f"Hausdorff Distance (HD): {hd:.4f}")
print(f"95% HD               : {hd95:.4f}")
print(f"Average Symmetric Surface Distance (ASSD): {assd:.4f}")