import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import color
from tqdm import tqdm
import math
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_image_and_mask(config):
    input_cfg = config['input']
    data_cfg = config['data']
    output_cfg = config['output']

    IMG_HEIGHT, IMG_WIDTH = input_cfg['resize_shape']
    TRAIN_PATH = input_cfg['train_path']
    IMAGE_PATH = os.path.join(TRAIN_PATH, input_cfg['image_dir'])
    LABEL_PATH = os.path.join(TRAIN_PATH, input_cfg['label_dir'])
    IMAGE_FORMAT = input_cfg.get('image_format', '.png')
    LOAD_FRACTION = input_cfg.get('load_fraction', 1.0)
    BATCH_SIZE = data_cfg['batch_size']
    SEED = data_cfg['random_seed']

    np.random.seed(SEED)

    train_ids = sorted([f for f in os.listdir(IMAGE_PATH) if f.endswith(IMAGE_FORMAT)])
    total = len(train_ids)
    num_to_load = math.ceil(total * LOAD_FRACTION)

    savefile_path = f"{output_cfg['savefile_path_prefix']}{BATCH_SIZE}{output_cfg['data_from']}"
    os.makedirs(savefile_path, exist_ok=True)

    X = np.zeros((num_to_load, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
    Y = np.zeros((num_to_load, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool_)

    print("Loading and resizing images and masks...")

    for n, fname in tqdm(enumerate(train_ids[:num_to_load]), total=num_to_load):
        image_path = os.path.join(IMAGE_PATH, fname)
        label_path = os.path.join(LABEL_PATH, fname.replace(IMAGE_FORMAT, '.png'))

        image = imread(image_path)
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[n] = image

        label_img = imread(label_path)
        label_img = resize(label_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        gray = color.rgb2gray(label_img[:, :, :3])
        binary = np.where(gray > 0, 255, 0)
        Y[n] = binary

    indices = np.arange(num_to_load)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    print(f"Loaded {num_to_load} samples.")
    return X, Y, savefile_path