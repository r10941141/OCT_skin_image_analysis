import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from tensorflow import keras
from keras import backend as K
import time
import argparse
import gc
import cv2
import math


parser = argparse.ArgumentParser(description='Calculate accuracy of the skin mask')
parser.add_argument('--epoch1', default=150, type=int, help='epoch of the mechine learning')
parser.add_argument('--splits_n', default=10, type=int, help='K fold of the mechine learning')
parser.add_argument('--batch_size', default=32, type=int, help='batch size of the mechine learning')
parser.add_argument('--savefile_path', default='/Workspace/datafile_batchsize', type=str, help='savefile path ')
parser.add_argument('--TRAIN_PATH', default='/Workspace/train/', type=str, help='TRAIN path ')
parser.add_argument('--TEST_PATH', default='/Workspace/test/', type=str, help='TEST path ')
parser.add_argument('--data_from', default='Andy', type=str, help='data_from')
parser.add_argument('--pretrain_model_path', default='/Workspace/datafile_CVproject_0607_16Andy/ML_for__pupil4.hdf5', type=str, help='')
args = parser.parse_args()


time_read_start = time.time()
seed = 46
np.random.seed = seed
IMG_WIDTH = 512
IMG_HEIGHT = 512

from ast import Num
batch_size_int=args.batch_size
trainer = args.data_from 
savefile_path =args.savefile_path + str(batch_size_int) + '_' + trainer
TRAIN_PATH = args.TRAIN_PATH
TEST_PATH = args.TEST_PATH
pretrain_model_path = args.pretrain_model_path

if not os.path.isdir(savefile_path):
    os.mkdir(savefile_path)

train_ids = next(os.walk(TRAIN_PATH))[2]
test_ids = next(os.walk(TEST_PATH))[2]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_ 
    if (path[-3::]) == 'pgm':
      name1=path[:len(TRAIN_PATH)]
      name2=path[len(TRAIN_PATH):-8]
      m1=path[-8:-4]
      mth=int(m1)
      mth=mth/4
      m2=str(mth)
      while len(m2)<6:
        m2='0'+m2
      m3=path[:-8] + m2[:-2] + '.png'

      #print(path)
      img = imread(path)[:,:]  
      #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
      X_train[n] = img  #Fill empty X_train with values from img
    
    #for mask_file in next(os.walk(path + '/masks/'))[2]:
      mpath = name1 + "mask/" + name2 + m2[:-2] + '.png'
      mask_ = imread(mpath)
      mask_=np.where(mask_<128, 0, mask_)
      #for k in range(IMG_HEIGHT):
      #  for i in range(IMG_WIDTH):
      #    if mask_[k][i]<128 :
      #      mask_[k][i] = 0
      #mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #mask = np.maximum(mask, mask_)  
      #print(m3)
      Y_train[n] = mask_


# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
sizes_test = []
    
num_samples = len(X_train)
indices = np.arange(num_samples)
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
X_train = X_train[:math.ceil(len(X_train)/1)]
Y_train = Y_train[:math.ceil(len(Y_train)/1)]

print('Resizing test images') 
for n ,id_ in enumerate(test_ids):
    path =TEST_PATH + id_
    img = imread(path)[:,:]
    #image =imread(path + '/images/' +id_ + '.pgm')[:,:] 
    sizes_test.append([img.shape[0], img.shape[1]])
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
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

      #print(path)
      img = imread(path)[:,:]  
      #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
      X_test[n] = img  #Fill empty X_train with values from img
    
    #for mask_file in next(os.walk(path + '/masks/'))[2]:
      mpath = name1 + "mask/" + name2 + m2[:-2] + '.png'
      test_mask = imread(mpath)
      mask_=np.where(mask_<128, 0, mask_)
      #  for i in range(IMG_WIDTH):
      #    if test_mask[k][i] <128 :
      #      test_mask[k][i] = 0
      #mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #mask = np.maximum(mask, mask_)  
    #  print(m3)
      Y_test[n] = test_mask
time_read_end = time.time()
time_imread = time_read_end - time_read_start
print('time read',time_imread , 's')
print('Done!')
'''
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
    '''

def dice(y_true,y_pred):
    #y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    dice = true_positives*2 / (predicted_positives + all_positives + K.epsilon())
    return dice

    #y_true = K.ones_like(y_true)
    '''
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def iou(y_true,y_pred):    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    iou = true_positives / (predicted_positives + all_positives - true_positives + K.epsilon())
    return iou
'''
"""# U-net建置"""


time_start = time.time()

def generator(X_train, Y_train, batch_size):
    num_samples = len(X_train)
    start = 0
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    while True:
        end = min(start + batch_size, (num_samples))

        x_batch = X_train[indices[start:end]]
        y_batch = Y_train[indices[start:end]]
        augmented_x_batch = np.zeros_like(x_batch)
        augmented_y_batch = np.zeros_like(y_batch)
        
        for i in range(x_batch.shape[0]):

            expanded_x = np.expand_dims(x_batch[i], axis=-1)
            expanded_y = np.expand_dims(y_batch[i], axis=-1)
            
            
            #augmented_x = adjust_brightness(expanded_x)           
            #augmented_x = adjust_contrast(augmented_x)
            #augmented_x, augmented_y = skew_image(expanded_x, expanded_y)
            #augmented_x = add_noise(augmented_x)
 
            ##if np.ndim(augmented_x) > 1:
            augmented_x = np.squeeze(expanded_x, axis=-1)
            augmented_y = np.squeeze(expanded_y , axis=-1)
            
            augmented_x_batch[i] = augmented_x
            augmented_y_batch[i] = augmented_y
       
        yield augmented_x_batch, augmented_y_batch
        start += batch_size
        if start >= (num_samples):
            start = 0
            np.random.shuffle(indices)
            
from sklearn.model_selection import KFold


def skew_image(image, mask):
    angle_min = np.deg2rad(-10)  # 最小角度 
    angle_max = np.deg2rad(10)  # 最大角度 
    skew_factor = np.random.uniform(angle_min, angle_max)
    factor = np.random.uniform(0, 100)
    image_shape = image.shape
    height = image_shape[0]
    width = image_shape[1]
    def apply_skew_r(img):
        rows, cols = img.shape[:2]
        M = cv2.getAffineTransform(np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1]]), np.float32([[0, rows * skew_factor], [cols - 1, 0], [cols - 1, rows - 1]]))
        skewed_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
        return skewed_img
    def apply_skew_l(img): 
        rows, cols = img.shape[:2]
        M = cv2.getAffineTransform(np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]]), np.float32([[0, 0], [cols - 1, rows * skew_factor], [0, rows - 1]]))
        skewed_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
        return skewed_img
    def apply_skew_u(img): 
        rows, cols = img.shape[:2]
        M = cv2.getAffineTransform(np.float32([[cols - 1, 0], [cols - 1, rows - 1], [0, 0]]), np.float32([[cols - 1, 0], [cols - 1, rows * (1 - skew_factor)], [0, 0]])) ##往上壓縮
        skewed_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
        return skewed_img
    def apply_skew_d(img): 
        rows, cols = img.shape[:2]
        M = cv2.getAffineTransform(np.float32([[cols - 1, 0], [cols - 1, rows - 1], [0, 0]]), np.float32([[cols - 1, rows* skew_factor], [cols - 1, rows -1], [0,           rows*skew_factor]])) #往下壓縮
        skewed_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
        return skewed_img
    
    mask = np.array(mask)
    image = np.array(image)
    image = image.astype(np.uint16)
    mask = 255 * np.array(mask).astype('uint8')
    
    if factor <= 25:     
      skewed_image = apply_skew_r(image)
      skewed_mask = apply_skew_r(mask.astype(np.uint8))
    elif factor > 25 and factor <= 50 :     
      skewed_image = apply_skew_l(image)
      skewed_mask = apply_skew_l(mask.astype(np.uint8))  
    elif factor > 50 and factor <= 75 :     
      skewed_image = apply_skew_u(image)
      skewed_mask = apply_skew_u(mask.astype(np.uint8)) 
    else:
      skewed_image = apply_skew_d(image)
      skewed_mask = apply_skew_d(mask.astype(np.uint8)) 
  

    skewed_mask = tf.cast(skewed_mask > 127, dtype=tf.bool)
    skewed_image= tf.cast(skewed_image, dtype=tf.uint16)
    
    return skewed_image, skewed_mask

def add_noise(image):
    m1 = np.random.uniform(0,10000)
    s1 = np.random.uniform(0,10000)
    noise = tf.random.normal(shape=tf.shape(image), mean=m1, stddev=s1, dtype=tf.float32)
    augmented_image = tf.cast(image, dtype=tf.float32) + noise
    augmented_image = tf.clip_by_value(augmented_image, 0, 65535)  
    augmented_image = tf.cast(augmented_image, dtype=tf.uint16)
    return augmented_image

def adjust_brightness(image):
    delta = np.random.uniform(-0.2, 0.2)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    augmented_image = tf.image.adjust_brightness(image_tensor, delta)
    augmented_image = tf.cast(augmented_image, dtype=tf.uint16)
    return augmented_image

def adjust_contrast(image):
    factor = np.random.uniform(0.8, 1.2) 
    augmented_image = tf.image.adjust_contrast(image, factor)
    return augmented_image

epoch1=args.epoch1
maxdice=0
splits_n=args.splits_n
fold_n=0
fold_dice=np.empty(splits_n)
#pretrained_model = tf.keras.models.load_model(pretrain_model_path, custom_objects={'dice': dice})
with open(savefile_path + '/tain_id.txt', "w") as file:
  file.write(str(train_ids))
with open(savefile_path + '/test_id.txt', "w") as file:
  file.write(str(test_ids))

fk = KFold(n_splits=splits_n, shuffle=True)
for trn, tst in fk.split(X_train) :
    
  fold_n=fold_n+1
  print(trn)
  print(tst)
  print("---")
  trn1=trn.size 
  tst1=tst.size 
  xtrain = np.zeros((trn1, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
  ytrain = np.zeros((trn1, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool_) 
  xtest = np.zeros((tst1, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
  ytest = np.zeros((tst1, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool_) 
  for k in range(trn1)  :
    xtrain[k]=(X_train[trn[k]])
    ytrain[k]=(Y_train[trn[k]])
  for k in range(tst1):
    xtest[k]=(X_train[tst[k]])
    ytest[k]=(Y_train[tst[k]])
    
  batch_size1=args.batch_size
    
  train_generator = generator(xtrain, ytrain, batch_size1)    
  steps_per_epoch = trn.size // args.batch_size
  validation_steps = tst.size // args.batch_size
    
  inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, 1)) 
  s = tf.keras.layers.Lambda(lambda x: x / 10000)(inputs)
  print('lambda',s)

  #Contraction path
  c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
  c1 = tf.keras.layers.Dropout(0.1)(c1)
  c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
  p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
  c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
  c2 = tf.keras.layers.Dropout(0.1)(c2)
  c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
  p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
  
  c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
  c3 = tf.keras.layers.Dropout(0.2)(c3)
  c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
  p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
  
  c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
  c4 = tf.keras.layers.Dropout(0.2)(c4)
  c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
  p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
  
  c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
  c5 = tf.keras.layers.Dropout(0.3)(c5)
  c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

  #Expansive path 
  u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
  u6 = tf.keras.layers.concatenate([u6, c4])
  c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
  c6 = tf.keras.layers.Dropout(0.2)(c6)
  c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
  
  u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
  u7 = tf.keras.layers.concatenate([u7, c3])
  c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
  c7 = tf.keras.layers.Dropout(0.2)(c7)
  c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
  
  u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
  u8 = tf.keras.layers.concatenate([u8, c2])
  c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
  c8 = tf.keras.layers.Dropout(0.1)(c8)
  c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
  
  u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
  u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
  c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
  c9 = tf.keras.layers.Dropout(0.1)(c9)
  c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


# 回調函數，若模型達到條件提早結束訓練
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')


  outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
  
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  #model.set_weights(pretrained_model.get_weights())
  opt = tf.keras.optimizers.Adam(learning_rate=0.0004)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', dice])
  
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir='logs')]
  
  results = model.fit(train_generator, validation_data=(xtest,ytest), steps_per_epoch=steps_per_epoch, epochs=args.epoch1, callbacks=[callbacks, early_stopping])
  stopped_epoch = early_stopping.stopped_epoch
  model.save(savefile_path + '/ML_for_skin_'+ str(fold_n) +'.hdf5')
  np.save(savefile_path + '/my_history' + str(fold_n) + '.npy', results.history)
  np.save(savefile_path + '/train_set'+ str(fold_n) + '.npy', trn)
  np.save(savefile_path + '/validation_set'+ str(fold_n) + '.npy', tst)
    
  if results.history['val_dice'][stopped_epoch-1]>maxdice:
    maxdice=results.history['val_dice'][stopped_epoch-1]
    model.save(savefile_path + '/ML_for_skin.hdf5')
    np.save(savefile_path + '/maxdice_train_set.npy', trn)
    np.save(savefile_path + '/maxdice_validation_set.npy', tst)
    
  K.clear_session()
  gc.collect()
  del model
  print('clear!')
print(maxdice)

####################################
time_end = time.time()
time_c = time_end - time_start
print('time read',time_imread , 's')
print('time cost',time_c , 's')
with open(savefile_path + '/time.txt', "w") as file:
  file.write(str(time_c))