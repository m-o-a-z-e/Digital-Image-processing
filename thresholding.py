"""
thresholding.py

This module provides functions for image thresholding using OpenCV and NumPy.
It includes basic global thresholding, automatic (Otsu's) thresholding, adaptive thresholding,
and vertical adaptive Otsu thresholding. All functions operate on NumPy image arrays and return
the thresholded (binary) image.
made by [ِAbdelrahman Salah]
"""

import cv2
import numpy as np

def Basic_Global_Thresholding(image, thresh_value):
    """Apply basic global thresholding to a grayscale version of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    return thresh

def Automatic_thresholding(image, thresh_value):
    """Apply Otsu's automatic thresholding to a grayscale version of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def Adaptive_thresholding(image):
    """Apply adaptive mean thresholding to a grayscale version of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

def vertical_adaptive_otsu(image):
    """Apply Otsu's thresholding to vertical segments of the grayscale image and concatenate the results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    parts = [gray[i*height//4:(i+1)*height//4, :] for i in range(4)]
    segmented = [cv2.threshold(p, 0, 255, cv2.THRESH_OTSU)[1] for p in parts]
    return np.concatenate(segmented, axis=0)
