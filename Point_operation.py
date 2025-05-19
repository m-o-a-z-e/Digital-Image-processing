"""
Point_operation.py

This module provides basic point operations for image processing using OpenCV and NumPy.
It includes functions to load an image, add, subtract, and divide pixel values, and compute the complement of an image.
All functions operate on NumPy image arrays and return the processed image.
made by [Nour Mohamed]
"""

import cv2
import numpy as np

def load_image(image_path):
    """Load an image from the specified file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}.")
    return image

def add_value_to_image(image, value):
    """Add a constant value to all pixels in the image."""
    return cv2.add(image, value)

def subtract_value_from_image(image, value):
    """Subtract a constant value from all pixels in the image."""
    return cv2.subtract(image, value)

def divide_image_by_value(image, value):
    """Divide all pixels in the image by a constant value."""
    if value == 0:
        raise ValueError("Division by zero is not allowed.")
    return cv2.divide(image, value)

def complement_image(image):
    """Compute the complement (negative) of the image."""
    return cv2.bitwise_not(image)
