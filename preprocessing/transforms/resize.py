import tensorflow as tf

def resize_image_and_mask(image, mask, target_size=(512, 512)):
    image = tf.image.resize(image, target_size)
    mask = tf.image.resize(tf.cast(mask, tf.float32), target_size)
    mask = tf.cast(mask > 0.5, tf.bool)
    return tf.cast(image, tf.uint16), mask