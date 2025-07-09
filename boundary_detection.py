import numpy as np
import cv2
from scipy.signal import medfilt2d
from skimage.restoration import denoise_wavelet
from skimage.transform import resize
import matplotlib.pyplot as plt

def wavelet_denoise_and_filter(image):
    denoised = denoise_wavelet(image, wavelet='sym4', method='BayesShrink', rescale_sigma=True)
    filtered = medfilt2d(denoised, kernel_size=(15, 15))
    Gy = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    return Gy

def find_surface_boundary(Gy, y_start=10, y_end=250):
    h, w = Gy.shape
    boundary = np.zeros(w, dtype=np.int32)
    for x in range(w):
        region = Gy[y_start:y_end, x]
        idx = np.argmax(np.abs(region))
        boundary[x] = idx + y_start
    return boundary

def fine_edj_boundary(boundary, Gy):
    smoothed = smooth_boundary_surface(boundary)
    h, w = Gy.shape
    final = np.zeros_like(smoothed)

    for x in range(w):
        q = smoothed[x]
        start = q + 20
        end = q + 48

        if end >= h:
            end = h - 1
        if start >= h:
            final[x] = h - 1 
            continue
        
        region = Gy[start:end, x]
        if region.size == 0:
            final[x] = start
        else:
            idx = np.argmax(np.abs(region))
            final[x] = start + idx

    return final

def smooth_boundary_surface(boundary):
    smoothed = boundary.copy()
    w = len(boundary)
    for x in range(6, w - 10):
        if abs(smoothed[x-1] - smoothed[x]) > 10:
            smoothed[x] = (smoothed[x-3] + smoothed[x+3]) // 2
        elif abs(smoothed[x-3] - smoothed[x]) > 20:
            smoothed[x] = (smoothed[x-5] + smoothed[x+5]) // 2
    for x in range(1, 50):
        if abs(smoothed[x] - smoothed[min(x+10, w-1)]) > 30:
            smoothed[x] = smoothed[min(x+20, w-1)]
    return smoothed


def smooth_boundary_edj(boundary):
    smoothed = boundary.copy()
    w = len(boundary)
    for x in range(6, w - 10):
        if abs(smoothed[x-1] - smoothed[x]) > 5:
            smoothed[x] = (smoothed[x-3] + smoothed[x+3]) // 2
        elif abs(smoothed[x-3] - smoothed[x]) > 15:
            smoothed[x] = (smoothed[x-3] + smoothed[x+10]) // 2
        elif abs(smoothed[x-3] - smoothed[x]) > 20:
            smoothed[x] = (smoothed[x-5] + smoothed[x+5]) // 2
    return smoothed



def show_boundary(image, boundary):
    plt.imshow(image, cmap='gray')
    plt.plot(np.arange(len(boundary)), boundary, color='r')
    plt.title("Detected Boundary")
    plt.show()