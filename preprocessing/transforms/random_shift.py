import numpy as np

def random_shift(image, mask, max_shift=50):
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    image = np.roll(image, shift=(shift_x, shift_y), axis=(0, 1))
    mask = np.roll(mask, shift=(shift_x, shift_y), axis=(0, 1))
    return image, mask