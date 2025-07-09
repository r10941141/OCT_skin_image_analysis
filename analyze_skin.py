from scipy.signal import medfilt2d
import cv2
import numpy as np


def calc_thickness(X, Y, pixel_size_um, n_epi):
    raw = np.abs(X - Y)
    sorted_thickness = np.sort(raw)[20:-20]  
    thickness_um = sorted_thickness * pixel_size_um / n_epi
    return thickness_um


def calc_smoothness(boundary, pixel_size_um, n_epi, fitting_order=3):

    x = np.arange(80, 866)
    y = boundary[80:866]
    p = np.polyfit(x, y, fitting_order)
    y_fit = np.polyval(p, x)
    diff = np.abs(y_fit - y)
    diff = np.sort(diff)[20:-20]
    diff_um = diff * pixel_size_um / n_epi
    return diff_um


def calc_OAC_mean(a2, X, Y):
    ep_u = []
    dm_u = []
    for x in range(944):
        x_epi = int(X[x])
        x_der = int(Y[x])
        if x_epi < x_der and x_der + 80 < 370:
            ep_region = a2[x_epi:x_der, x]
            dm_region = a2[x_der:x_der + 80, x]
            if len(ep_region) > 0:
                ep_u.append(np.mean(ep_region))
            if len(dm_region) > 0:
                dm_u.append(np.mean(dm_region))
    
    avg_ep = np.mean(ep_u) * 1e-3 if ep_u else 0
    avg_dm = np.mean(dm_u) * 1e-3 if dm_u else 0
    return avg_ep, avg_dm


def analyze_OCT_image(a2, X, Y, pixel_size_um=3.3, n_epi=1.4):

    thickness_um = calc_thickness(X, Y, pixel_size_um, n_epi)

    surface_fit_error = calc_smoothness(X, pixel_size_um, n_epi)
    edj_fit_error = calc_smoothness(Y, pixel_size_um, n_epi)

    ep_u, dm_u = calc_OAC_mean(a2, X, Y)

    return {
        "thickness_mean": (thickness_um),
        "surface_smooth_mean": (surface_fit_error),
        "edj_smooth_mean": (edj_fit_error),
        "ep_u_mean": (ep_u) ,
        "dm_u_mean": (dm_u) 
    }