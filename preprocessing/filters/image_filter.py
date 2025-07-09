import tensorflow as tf
import numpy as np

def adjust_brightness(image, delta_range=(-0.2, 0.2)):
    delta = tf.random.uniform([], *delta_range)
    return tf.image.adjust_brightness(image, delta)

def adjust_contrast(image, delta_range=(0.8, 1.2)):
    factor = tf.random.uniform([], *delta_range)
    return tf.image.adjust_contrast(image, factor)

def add_noise(image):
    m1 = np.random.uniform(0,10000)
    s1 = np.random.uniform(0,10000)
    noise = tf.random.normal(shape=tf.shape(image), mean=m1, stddev=s1, dtype=tf.float32)
    augmented_image = tf.cast(image, dtype=tf.float32) + noise
    augmented_image = tf.clip_by_value(augmented_image, 0, 65535)  
    augmented_image = tf.cast(augmented_image, dtype=tf.uint16)
    return augmented_image


