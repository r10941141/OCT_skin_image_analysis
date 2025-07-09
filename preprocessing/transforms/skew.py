import numpy as np
import cv2
import tensorflow as tf

def skew_image(image, mask, angle_range(-10, 10)):

    angle_min = np.deg2rad(angle_range[0])  
    angle_max = np.deg2rad(angle_range[1]) 
    skew_factor = np.random.uniform(angle_min, angle_max)
    factor = np.random.uniform(0, 100)

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