import json 
from boundary_detection import find_surface_boundary, fine_edj_boundary, smooth_boundary_surface, smooth_boundary_edj
from analyze_skin import analyze_OCT_image
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
mode = main_config['mode']

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

img_resize = tf.image.resize(tf.cast(img, tf.float32), (370,944))
label_resize = tf.image.resize(tf.cast(label, tf.float32), (370,944))
C[0] = label
S[0] = img

if mode > 0 :
    module = importlib.import_module(f"model.{classification_training_config }")
    build_func = getattr(module, f"build_{classification_training_config }")
    model = build_func()  
    model.load_weights(load_model_path_c) 
    model.compile(optimizer=classification_training_config['optimizer'], loss = classification_training_config['loss'], metrics=['accuracy'])
    c_predictions = model.predict(C)

    if c_predictions < 0.8 and mode > 1 :
        model_s = tf.keras.models.load_model(load_model_path_s , custom_objects={ 'dice': dice})
        s_predictions = model_s.predict(S)
        s_predictions = np.squeeze(s_predictions)
        pre_mask = tf.image.resize(s_predictions, (370,944))
        surface = find_surface_boundary(pre_mask)
        surface = smooth_boundary_surface(surface)
        edj = fine_edj_boundary(surface, pre_mask)
        edj = smooth_boundary_edj(edj)
    else :
        surface = find_surface_boundary(label_resize)
        surface = smooth_boundary_surface(surface)
        edj = fine_edj_boundary(surface, label_resize)
        edj = smooth_boundary_edj(edj)
else :
    surface = find_surface_boundary(label_resize)
    surface = smooth_boundary_surface(surface)
    edj = fine_edj_boundary(surface, label_resize)
    edj = smooth_boundary_edj(edj)

result = analyze_OCT_image(img_resize, surface, edj)
print(result)