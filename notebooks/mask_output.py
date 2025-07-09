import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from skimage.io import imread, imshow
from keras import backend as K
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import math
import argparse
import os
import time
import cv2
import gc


parser = argparse.ArgumentParser(description='Calculate accuracy of the mask')
parser.add_argument('--TEST_PATH', default='/Workspace/images_andy/test/', type=str, help='file whitch target_file in')
parser.add_argument('--save_mask_path', default='/Workspace/pred_mask_testing_data_0628/', type=str, help='save_mask_path')
parser.add_argument('--model_path', default='/Workspace/datafile_batchsize16Andy/', type=str, help='model_path')
parser.add_argument('--splits_n', default=1, type=int, help='')
args = parser.parse_args()

TEST_PATH = args.TEST_PATH
save_mask_path = args.save_mask_path
model_path = args.model_path 
splits_n = args.splits_n

test_ids = next(os.walk(TEST_PATH))[2]
time_read_start = time.time()
seed = 46
np.random.seed = seed
IMG_WIDTH = 512
IMG_HEIGHT = 512
# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
sizes_test = []

print('Resizing test images') 
for n ,id_ in enumerate(test_ids):
    path =TEST_PATH + id_
    img = imread(path)[:,:]
    sizes_test.append([img.shape[0], img.shape[1]])
    X_test[n] = img 

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):   
    path = TEST_PATH + id_ 
    if (path[-3::]) == 'pgm':
      name1=path[:len(TEST_PATH)]
      name2=path[len(TEST_PATH):-8]
      m1=path[-8:-4]
      mth=int(m1)
      mth=mth/4
      m2=str(mth)
      while len(m2)<6:
        m2='0'+m2
      m3=path[:-8] + m2[:-2] + '.png'

      img = imread(path)[:,:]  
      X_test[n] = img  #Fill empty X_train with values from img


    
time_read_end = time.time()
time_imread = time_read_end - time_read_start
print('time read',time_imread , 's')
print('Done!')


#定義模型參數
def recall(y_true,y_pred):                          
    #y_true = K.ones_like(y_true)                              
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true,y_pred):
    #y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def dice(y_true,y_pred):
    #y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    dice = true_positives*2 / (predicted_positives + all_positives + K.epsilon())
    return dice
    #y_true = K.ones_like(y_true)
    


if not os.path.isdir(save_mask_path):
    os.mkdir(save_mask_path)
    
save_mask_path02 = save_mask_path +  'mask1024/'   
if not os.path.isdir(save_mask_path02):
    os.mkdir(save_mask_path02)

def generator(X_train, batch_size):
    num_samples = len(X_train)
    start = 0
    indices = np.arange(num_samples)
    while True:
        end = min(start + batch_size, (num_samples))
        x_batch = X_train[indices[start:end]]
        yield x_batch
        start += batch_size    
        
batch_size = 16
gen = generator(X_test, batch_size)

save_mask_path_02 = save_mask_path + 'mask1024/'
if not os.path.isdir(save_mask_path_02):
    os.mkdir(save_mask_path_02)
    
L1 = math.ceil(len(test_ids) / batch_size)
for i in tqdm(range(L1)):
    x_batch = next(gen)
    preds_oac_mask = 0
    for k in range(splits_n):        #splits_n需定義
        model = tf.keras.models.load_model(model_path + 'ML_for_skin_' + str(k+1) +'.hdf5', custom_objects={ 'dice': dice})
        preds_oac_mask = preds_oac_mask + model.predict(x_batch, verbose=1)/splits_n
    preds_oac_mask_t = (preds_oac_mask > 0.5).astype(np.uint8)
    preds_oac_mask = np.where(preds_oac_mask_t > 0.5,255,0)
    K.clear_session()
    gc.collect()
    del model
    for j, mask in enumerate(preds_oac_mask):
        plt.imsave(save_mask_path + test_ids[i * batch_size + j][:-4] + '.png', np.squeeze(mask*255), vmin=0, vmax=255, cmap='gray', format='png')
        img = Image.open(save_mask_path + test_ids[i * batch_size + j][:-4] + '.png').convert('L')
        img.save(save_mask_path + test_ids[i * batch_size + j][:-4]  + '.png')

        img = cv2.imread(save_mask_path + test_ids[i * batch_size + j][:-4] + '.png' ,cv2.IMREAD_GRAYSCALE)
        resized = (cv2.resize(img, (944, 370), interpolation = cv2.INTER_LINEAR))
        cv2.imwrite(save_mask_path + 'mask1024/' + test_ids[i * batch_size + j][:-4]  + '.png'  , resized)
###  
