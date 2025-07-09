# train.py

import os
import gc
import json
import math
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import KFold
from skimage.io import imread
from skimage.transform import resize
from skimage import color
from tensorflow.keras.callbacks import TensorBoard
import importlib
from metrics.confusion_metrics import dice, recall, precision
from data_generator import generator

# Load configs
with open('./config/config_data_loading.json') as f:
    data_config_all = json.load(f)
with open('./config/config_preprocessing.json') as f:
    aug_config = json.load(f)['preprocessing']
with open('./config/config_training.json') as f:
    training_config_all = json.load(f)['segmentation_training']
with open('./config/config_model.json') as f:
    config_model_all = json.load(f)


data_config = data_config_all["segmentation_training"]
training_config = training_config_all["segmentation_training"]

# Constants from config
model_used = training_config['use_model']
train_path = data_config['input']['train_path']
image_dir = os.path.join(train_path, data_config['input']['image_dir'])
label_dir = os.path.join(train_path, data_config['input']['label_dir'])
resize_shape = tuple(data_config['input']['resize_shape'])
batch_size = training_config['batch_size']
load_fraction = data_config['input']['load_fraction']
seed = data_config['data']['random_seed']
kfold_n = training_config['k_fold']
epochs = training_config['epochs']
savefile_path = data_config['output']['savefile_path_prefix'] + str(batch_size) + data_config['output']['data_from']
config_model = config_model_all.get(model_used)
if config_model is None:
    raise ValueError(f"No config found for model: {model_used}")

np.random.seed(seed)
random.seed(seed)

# Create save path
os.makedirs(savefile_path, exist_ok=True)

# Load data
train_ids = sorted(os.listdir(image_dir))
num_samples = math.ceil(len(train_ids) * load_fraction)
train_ids = train_ids[:num_samples]

X = np.zeros((len(train_ids), *resize_shape), dtype=np.uint16)
Y = np.zeros((len(train_ids), *resize_shape), dtype=bool)

print("Loading and resizing training data")
for i, file_id in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(os.path.join(image_dir, file_id))
    img = resize(img, resize_shape, mode='constant', preserve_range=True)
    X[i] = img

    label_path = os.path.join(label_dir, file_id)
    label = imread(label_path)
    label = resize(label, resize_shape, mode='constant', preserve_range=True)
    gray = color.rgb2gray(label[:, :, :3]) if label.ndim == 3 else label
    Y[i] = np.where(gray > 0, 1, 0)

indices = np.arange(len(X))
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

# K-Fold
kf = KFold(n_splits=kfold_n, shuffle=True, random_state=seed)
max_dice = 0
fold_dice_scores = []

module = importlib.import_module(f"model.{model_used}")
build_func = getattr(module, f"build_{model_used}")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1}/{kfold_n} ---")

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    train_gen = generator(X_train, Y_train, batch_size, aug_config)

    model = build_func(config=config_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config['optimizer']['learning_rate']),
        loss=training_config['loss'],
        metrics=['accuracy', precision, recall, dice]
    )

    history = model.fit(
        train_gen,
        validation_data=(np.expand_dims(X_val, -1), np.expand_dims(Y_val, -1)),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[TensorBoard(log_dir='logs')]
    )

    fold_dice = history.history['val_dice'][-1]
    fold_dice_scores.append(fold_dice)

    model.save(f"{savefile_path}/{model_used}_fold{fold+1}.hdf5")
    np.save(f"{savefile_path}/{model_used}_history_fold{fold+1}.npy", history.history)

    if fold_dice > max_dice:
        max_dice = fold_dice
        model.save(f"{savefile_path}/{model_used}_best.hdf5")
        np.save(f"{savefile_path}/{model_used}_best.npy", history.history)
    tf.keras.backend.clear_session()
    gc.collect()

print(f"\nMax dice score across folds: {max_dice:.4f}")

