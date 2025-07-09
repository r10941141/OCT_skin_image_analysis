import json
import importlib
import os
import numpy as np
import math
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize


with open('./config/config_training.json') as f:
    training_config_all = json.load(f)['classification_training']
with open('./config/config_model.json') as f:
    config_model_all = json.load(f)
with open('./config/config_data_loading.json') as f:
    data_config_all = json.load(f)

data_config = data_config_all["classification_training"]
training_config = training_config_all["classification_training"]
load_fraction = data_config['input']['load_fraction']
resize_shape = tuple(data_config['input']['resize_shape'])
savefile_path = data_config['output']['savefile_path_prefix']  + data_config['output']['data_from']
os.makedirs(savefile_path, exist_ok=True)

train_path_c = data_config['input']['train_path_correct']
image_dir_c = os.path.join(train_path_c, data_config['input']['image_dir'])
train_ids = sorted(os.listdir(image_dir_c))
num_samples = math.ceil(len(train_ids) * load_fraction)
train_ids_c = train_ids[:num_samples]
x_train_c = np.zeros((len(train_ids), *resize_shape), dtype=np.uint16)
Y_pass_train_c = np.full((len(train_ids_c), 1), 1, dtype=np.float32)

for i, file_id in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(os.path.join(image_dir_c, file_id))
    img = resize(img, resize_shape, mode='constant', preserve_range=True)
    x_train_c[i] = img



train_path_w = data_config['input']['train_path_wrong']
image_dir_w = os.path.join(train_path_w, data_config['input']['image_dir'])
train_ids = sorted(os.listdir(image_dir_w))
num_samples = math.ceil(len(train_ids) * load_fraction)
train_ids_w = train_ids[:num_samples]
x_train_w = np.zeros((len(train_ids), *resize_shape), dtype=np.uint16)
Y_pass_train_w = np.full((len(train_ids_w), 1), 0, dtype=np.float32)

for i, file_id in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(os.path.join(image_dir_w, file_id))
    img = resize(img, resize_shape, mode='constant', preserve_range=True)
    x_train_w[i] = img



model_used = training_config['use_model']
config_model = config_model_all.get(model_used)
if config_model is None:
    raise ValueError(f"No config found for model: {model_used}")
module = importlib.import_module(f"model.{model_used}")
build_func = getattr(module, f"build_{model_used}")
Y_pass_train = np.concatenate((Y_pass_train_c, Y_pass_train_w), axis=0)
X_train = np.concatenate((x_train_c, x_train_w), axis=0)
X_train = np.expand_dims(X_train, axis=-1)

model = build_func()


model.compile(optimizer=training_config['optimizer'], loss = training_config['loss'], metrics=['accuracy'])


history = model.fit(x=X_train, 
                    y=Y_pass_train, 
                    batch_size = training_config["batch_size"], 
                    epochs = training_config["epochs"],
                    validation_split = training_config["validation_split"])


model.save(f"{savefile_path}/{model_used}.h5")
np.save(f"{savefile_path}/{model_used}_best.npy", history.history)