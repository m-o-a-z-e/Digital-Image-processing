"""
Image_Restoration_ed.py

This module provides functions for simulating and restoring images with noise using OpenCV and NumPy.
It includes functions to add salt-and-pepper noise, add Gaussian noise, and apply various filters
(outlier, average, median) for noise reduction and image restoration.
Each function operates on a NumPy image array and returns the processed image.
made by [Nour Mohamed]
"""

import cv2
import numpy as np

# Adds salt-and-pepper noise to the image with a given probability
def add_salt_pepper_noise(image, prob=0.02):
    noisy = image.copy()
    probs = np.random.rand(*image.shape[:2])
    noisy[probs < prob / 2] = 0
    noisy[probs > 1 - prob / 2] = 255
    return noisy

# Adds Gaussian noise to the image with specified mean and standard deviation
def add_gaussian_noise(image, mean=0, std=20):
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), gauss)
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Applies an outlier filter to reduce noise by replacing outlier pixels with the local mean
def outlier_filter(img, threshold=30):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = img.copy()

    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            window = img_padded[i-1:i+2, j-1:j+2]
            center = float(img_padded[i, j])
            mean = float(np.mean(window))
            if abs(center - mean) > threshold:
                result[i-1, j-1] = int(mean)
    return result.astype(np.uint8)

# Applies a 3x3 average filter to smooth the image
def average_filter(image):
    h, w = image.shape
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.mean(window)
    return result.astype(np.uint8)

# Applies a 3x3 median filter to reduce noise in the image
def median_filter(image):
    h, w = image.shape
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.median(window)
    return result.astype(np.uint8)

# Averages a list of images to reduce noise (image averaging)
def image_averaging(img1, img2):
    avg = (img1.astype(np.float32) + img2.astype(np.float32)) / 2
    return avg.astype(np.uint8)
