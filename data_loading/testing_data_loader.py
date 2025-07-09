import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm

def load_test_data(image_dir, gt_mask_dir, pred_mask_dir, 
                   img_height=512, img_width=512):
    """
    Parameters:
        image_dir (str): Directory path for original test images.
        gt_mask_dir (str, optional): Directory path for ground truth masks.
        pred_mask_dir (str, optional): Directory path for predicted masks.
        img_height (int): Height of the images.
        img_width (int): Width of the images.

    Returns:
        tuple: (X_test, Y_test, Z_test), each as a NumPy array with shape (N, H, W).
               If gt_mask_dir or pred_mask_dir is not provided, the corresponding
               output will be None.
    """
    image_files = sorted([
        f for f in next(os.walk(image_dir))[2]
        if f.lower().endswith(('.png', '.pgm'))
    ])
    
    N = len(image_files)
    X_test = np.zeros((N, img_height, img_width), dtype=np.uint16)
    Y_test = np.zeros((N, img_height, img_width), dtype=np.uint16) #if gt_mask_dir else None
    Z_test = np.zeros((N, img_height, img_width), dtype=np.uint16) #if pred_mask_dir else None

    print("Loading original test images")
    for n, filename in enumerate(image_files):
        img_path = os.path.join(image_dir, filename)
        img = imread(img_path)
        X_test[n] = img

    if gt_mask_dir:
        print("Loading ground truth masks")
        for n, filename in tqdm(enumerate(image_files), total=N):
            mask_path = os.path.join(gt_mask_dir, filename)
            if os.path.exists(mask_path):
                mask = imread(mask_path)
                Y_test[n] = np.where(mask < 128, 0, mask)
            else:
                print("Ground truth mask not found")

    if pred_mask_dir:
        print("Loading predicted masks")
        for n, filename in tqdm(enumerate(image_files), total=N):
            mask_path = os.path.join(pred_mask_dir, filename)
            if os.path.exists(mask_path):
                mask = imread(mask_path)
                Z_test[n] = np.where(mask < 128, 0, mask)
            else:
                print("Predicted mask not found")

    return X_test, Y_test, Z_test