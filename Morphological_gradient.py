"""
Morphological_gradient.py

This module provides functions for performing basic morphological operations on binary images using OpenCV and NumPy.
It includes dilation, erosion, opening, internal and external boundary extraction, and morphological gradient.
All functions operate on NumPy image arrays and return the processed binary image.
made by [Hager Mostafa]
"""

import cv2
import numpy as np

def preprocess_binary(image):
    """Convert input image to binary."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary

def apply_dilation(image, kernel_size=(5, 5)):
    """Apply dilation to a binary image using a given kernel size."""
    binary = preprocess_binary(image)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)

def apply_erosion(image, kernel_size=(5, 5)):
    """Apply erosion to a binary image using a given kernel size."""
    binary = preprocess_binary(image)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(binary, kernel, iterations=1)

def apply_opening(image, kernel_size=(5, 5)):
    """Apply morphological opening to a binary image using a given kernel size."""
    binary = preprocess_binary(image)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

def internal_boundary(image, kernel_size=(5, 5)):
    """Extract the internal boundary of objects in a binary image."""
    binary = preprocess_binary(image)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.subtract(binary, cv2.erode(binary, kernel))

def external_boundary(image, kernel_size=(5, 5)):
    """Extract the external boundary of objects in a binary image."""
    binary = preprocess_binary(image)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.subtract(cv2.dilate(binary, kernel), binary)

def morphological_gradient(image, kernel_size=(5, 5)):
    """Compute the morphological gradient of a binary image."""
    binary = preprocess_binary(image)
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
