from .filters import image_filter
from .transforms import resize, random_shift, perspective
import tensorflow as tf

def apply_augmentation(image, mask=None):
    image = tf.expand_dims(image, axis=-1)
    image = image_filter.adjust_brightness(image)
    image = image_filter.adjust_contrast(image)
    image, mask = resize.resize_image_and_mask(image, mask)
    image, mask = random_shift.random_shift(image.numpy(), mask.numpy())
    image, mask = perspective.perspective_transform(image, mask)
    return tf.squeeze(image, axis=-1), mask

