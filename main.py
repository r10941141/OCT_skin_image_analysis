import json 
from boundary_detection import
from analyze_skin import 
from skimage.io import imread
from preprocessing.transforms import resize
from metrics.confusion_metrics import dice
import importlib
import tensorflow as tf
import numpy as np

with open('./config/config_main.json') as f:
    main_config = json.load(f)
with open('./config/config_training.json') as f:
    training_config_all = json.load(f)

segmentation_training_config = training_config_all["segmentation_model"]
classification_training_config = training_config_all["classification_model"]

input_img_path = main_config['input_img_path']
input_label_path = main_config['input_label_path']
load_model_path = main_config['load_model_path']

segmentation_model_used = segmentation_training_config['use_model'] 
classification_model_used = classification_training_config['use_model']


load_model_path_s = (f"{load_model_path}/{segmentation_model_used}_best.hdf5")
load_model_path_c = (f"{load_model_path}/{classification_model_used}.h5")

segmentation_model_used = segmentation_training_config['use_model']
classification_training_config = classification_training_config['use_model']

img = imread(input_img_path)
label = imread(input_label_path)
C = np.zeros((1, 512, 512, 1), dtype=bool)
S = np.zeros((1, 512, 512, 1), dtype=np.uint16)
img_resize, label_resize = resize.resize_image_and_mask(img, label)

module = importlib.import_module(f"model.{classification_training_config }")
build_func = getattr(module, f"build_{classification_training_config }")
model = build_func()  
model.load_weights(load_model_path_c) 
model.compile(optimizer=classification_training_config['optimizer'], loss = classification_training_config['loss'], metrics=['accuracy'])

model = tf.keras.models.load_model(load_model_path_s , custom_objects={ 'dice': dice})