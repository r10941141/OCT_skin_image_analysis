import numpy as np
import time
from tqdm import tqdm

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False


def cal_distance(p1, p2, xp):
    """Compute euclidean distance between points."""
    return xp.sqrt(xp.sum((p1 - p2) ** 2, axis=-1))


def evaluate_hd_metrics(Y_test_np, Z_test_np, img_height=512, img_width=512, use_gpu=True):
    """
    Evaluate Hausdorff Distance, 95% HD, and ASSD between prediction and ground truth.

    Parameters:
        Y_test_np (np.ndarray): Ground truth masks, shape (N, H, W)
        Z_test_np (np.ndarray): Predicted masks, shape (N, H, W)
        img_height (int): Image height (default: 512)
        img_width (int): Image width (default: 512)
        use_gpu (bool): Whether to use GPU acceleration via CuPy

    Returns:
        tuple: (HD, HD95, ASSD)
    """
    time_start = time.time()

    if use_gpu and not has_cupy:
        raise ImportError("CuPy is not installed, but GPU computation was requested.")

    xp = cp if use_gpu else np
    Y_test = xp.asarray(Y_test_np)
    Z_test = xp.asarray(Z_test_np)

    hdsum, hd95sum, assdsum = 0, 0, 0

    for q in tqdm(range(len(Z_test))):
        x2 = xp.zeros(img_width * 4, dtype=xp.float32)  
        x_plot = xp.arange(img_width) + 1

        # dermis boundary
        y_d = xp.abs(xp.argmax(xp.flip(Z_test[q]), axis=0) - (img_height - 1))
        y_d_gt = xp.abs(xp.argmax(xp.flip(Y_test[q]), axis=0) - (img_height - 1))
        # epidermis boundary
        y_e = xp.argmax(Z_test[q], axis=0)
        y_e_gt = xp.argmax(Y_test[q], axis=0)

        for k in range(img_width):
            # Dermis comparisons
            # Use stack to create (N, 2) coordinate arrays for all boundary points.
            # Enables parallel distance computation between one point and all others.
            x1 = cal_distance(xp.array([y_d_gt[k], x_plot[k]]), xp.stack((y_d, x_plot), axis=-1), xp)
            x2[k] = xp.min(x1)
            x1 = cal_distance(xp.array([y_d[k], x_plot[k]]), xp.stack((y_d_gt, x_plot), axis=-1), xp)
            x2[img_width + k] = xp.min(x1)
            # Epidermis comparisons
            x1 = cal_distance(xp.array([y_e_gt[k], x_plot[k]]), xp.stack((y_e, x_plot), axis=-1), xp)
            x2[img_width * 2 + k] = xp.min(x1)
            x1 = cal_distance(xp.array([y_e[k], x_plot[k]]), xp.stack((y_e_gt, x_plot), axis=-1), xp)
            x2[img_width * 3 + k] = xp.min(x1)

        x2 = xp.sort(x2)
        assdsum += xp.mean(x2)
        hdsum += xp.max(x2)
        hd95sum += x2[int(img_width * 3.8)]  # ≈95% 

    n = len(Z_test)
    hd = hdsum / n
    hd95 = hd95sum / n
    assd = assdsum / n

    time_end = time.time()
    print(f"Time cost: {time_end - time_start:.2f} s")

    return float(hd), float(hd95), float(assd)