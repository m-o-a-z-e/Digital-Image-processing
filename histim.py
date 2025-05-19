"""
histim.py

This module provides functions for histogram-based image processing using OpenCV and NumPy.
It includes functions for histogram stretching, histogram equalization (manual and OpenCV-based), 
and histogram computation for grayscale and color images.
Each function operates on a NumPy image array and returns the processed image or histogram.
made by [Moaz Hany]
"""

import cv2
import numpy as np

# Performs histogram stretching to enhance the contrast of the image
def hist_stretching(image):
    if len(image.shape) == 2:  # Grayscale
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val == min_val:
            return image.copy()
        stretched_image = (image - min_val) * (255.0 / (max_val - min_val))
        return stretched_image.astype(np.uint8)
    else:  # Color image
        stretched_image = np.zeros_like(image)
        for i in range(3):
            min_val = np.min(image[:, :, i])
            max_val = np.max(image[:, :, i])
            if max_val == min_val:
                stretched_image[:, :, i] = image[:, :, i]
            else:
                stretched_image[:, :, i] = (image[:, :, i] - min_val) * (255.0 / (max_val - min_val))
        return stretched_image.astype(np.uint8)

# Computes the histogram of a single channel (grayscale or color)
def compute_histogram(channel):
    hist = [0] * 256
    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            hist[channel[i, j]] += 1
    return hist

# Performs histogram equalization to improve the contrast of the image
def histogram_equalization(image):
    if len(image.shape) == 2:  # Grayscale
        hist = compute_histogram(image)
        cdf = [0] * 256
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + hist[i]
        cdf_min = next((v for v in cdf if v > 0), 0)
        total_pixels = cdf[-1]
        mapping = [round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255) if total_pixels != cdf_min else 0 for i in range(256)]
        h, w = image.shape
        equalized_image = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                equalized_image[i, j] = mapping[image[i, j]]
        return equalized_image
    else:  # Color image
        h, w, _ = image.shape
        equalized_image = np.zeros_like(image)
        for ch in range(3):
            hist = compute_histogram(image[:, :, ch])
            cdf = [0] * 256
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i-1] + hist[i]
            cdf_min = next((v for v in cdf if v > 0), 0)
            total_pixels = cdf[-1]
            mapping = [round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255) if total_pixels != cdf_min else 0 for i in range(256)]
            for i in range(h):
                for j in range(w):
                    equalized_image[i, j, ch] = mapping[image[i, j, ch]]
        return equalized_image

# Performs histogram equalization on each RGB channel using OpenCV's built-in function
def hist_equalize_rgb(image):
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    b, g, r = cv2.split(image)
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    return cv2.merge((b_eq, g_eq, r_eq))
