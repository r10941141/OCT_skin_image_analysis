import cv2
import numpy as np
import tensorflow as tf

def perspective_transform(image, mask, angle_deg_range=(-20, 65)):
    rows, cols = image.shape[:2]
    angle = np.deg2rad(np.random.uniform(*angle_deg_range))
    zz = 2 / ((np.sin(angle)) * 2.5 + 2)
    side = np.random.rand() > 0.5
    if side:
        dst = np.float32([[-(np.cos(angle)-0.5)*zz*cols + cols/2, cols/2*(1-zz)],
                          [cols-1, 0],
                          [-(np.cos(angle)-0.5)*zz*cols + cols/2, cols/2*(1+zz)],
                          [cols-1, rows-1]])
    else:
        dst = np.float32([[0, 0],
                          [(np.cos(angle)-0.5)*zz*cols + cols/2, cols/2*(1-zz)],
                          [0, rows-1],
                          [(np.cos(angle)-0.5)*zz*cols + cols/2, cols/2*(1+zz)]])
    src = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(image.astype(np.uint16), M, (cols, rows))
    mask = cv2.warpPerspective((mask * 255).astype(np.uint8), M, (cols, rows))
    mask = tf.cast(mask > 127, dtype=tf.bool)
    return image, mask
