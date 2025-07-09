from preprocessing.filters import image_filter
from preprocessing.transforms import resize, random_shift, perspective, skew
import numpy as np


def generator(X_train, Y_train, batch_size, aug_config ):
    num_samples = len(X_train)
    start = 0
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    
    while True:
        end = min(start + batch_size, num_samples)
        x_batch = X_train[indices[start:end]]
        y_batch = Y_train[indices[start:end]]

        augmented_x_batch = np.zeros_like(x_batch)
        augmented_y_batch = np.zeros_like(y_batch)
        
        for i in range(x_batch.shape[0]):
            x = np.expand_dims(x_batch[i], axis=-1)
            y = np.expand_dims(y_batch[i], axis=-1)

            if aug_config["enable_noise"]:
                x = image_filter.add_noise(x)

            if aug_config["enable_brightness_adjustment"]:
                x = image_filter.adjust_brightness(x, aug_config["brightness_delta_range"])

            if aug_config["enable_contrast_adjustment"]:
                x = image_filter.adjust_contrast(x, aug_config["contrast_range"])

            if aug_config["enable_resize"]:
                x, y = resize.resize_image_and_mask(x, y, aug_config["resize_target_size"])

            if aug_config["enable_random_shift"]:
                x, y = random_shift.random_shift(x, y, aug_config["random_shift_max"])
            
            if aug_config["enable_skew_transform"]:
                x, y = skew.skew_image(x, y, aug_config["skew_angle_range_deg"])


            if aug_config["enable_perspective_transform"]:
                x, y = perspective.perspective_transform(x, y, aug_config["perspective_angle_range_deg"])

            augmented_x_batch[i] = np.squeeze(x)
            augmented_y_batch[i] = np.squeeze(y)

        yield augmented_x_batch, augmented_y_batch
        
        start += batch_size
        if start >= num_samples:
            start = 0
            np.random.shuffle(indices)