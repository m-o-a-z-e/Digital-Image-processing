"""
filtersim.py

This module provides implementations of various spatial filters for image processing using NumPy and OpenCV.
It includes average, median, mode, maximum, minimum, and Laplacian filters, which can be applied to color images.
Each filter function takes an image as input and returns the filtered image.
made by [Moaz Hany]
"""

import cv2
import numpy as np

# Applies a 3x3 average (mean) filter to the image to smooth it
def apply_average_filter(image):
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                window = padded[i:i+3, j:j+3, c]
                output[i, j, c] = np.sum(window) // 9
    return output

# Applies a 3x3 Laplacian filter to detect edges in the image
def apply_laplacian_filter(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padded = np.pad(gray, 1, mode='edge')
    output = np.zeros_like(gray)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            region = padded[i:i+3, j:j+3]
            val = np.sum(region * kernel)
            output[i, j] = np.clip(abs(val), 0, 255)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

# Applies a 3x3 maximum filter to the image (dilation effect)
def apply_maximum_filter(image):
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                output[i, j, c] = np.max(padded[i:i+3, j:j+3, c])
    return output

# Applies a 3x3 minimum filter to the image (erosion effect)
def apply_minimum_filter(image):
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                output[i, j, c] = np.min(padded[i:i+3, j:j+3, c])
    return output

# Applies a 3x3 median filter to the image to reduce noise
def apply_median_filter(image):
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                window = padded[i:i+3, j:j+3, c].flatten()
                output[i, j, c] = np.median(window)
    return output

# Applies a 3x3 mode filter to the image (most frequent value in the window)
def apply_mode_filter(image):
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                window = padded[i:i+3, j:j+3, c].flatten()
                counts = np.bincount(window)
                output[i, j, c] = np.argmax(counts)
    return output

# Applies the selected filter type to the image
def apply_filter(image, filter_type="median"):
    if filter_type == "median":
        return apply_median_filter(image)
    elif filter_type == "average":
        return apply_average_filter(image)
    elif filter_type == "laplacian":
        return apply_laplacian_filter(image)
    elif filter_type == "maximum":
        return apply_maximum_filter(image)
    elif filter_type == "minimum":
        return apply_minimum_filter(image)
    elif filter_type == "mode":
        return apply_mode_filter(image)
    return image
