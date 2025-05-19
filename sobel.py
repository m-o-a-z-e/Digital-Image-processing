"""
sobel.py

This module provides functions for edge detection using the Sobel operator with OpenCV and NumPy.
It includes a custom convolution function and applies the Sobel operator in both X and Y directions.
The main function returns the original image, the combined Sobel edge image, and the gradients in X and Y.
made by [Hager Mostafa]
"""

import cv2
import numpy as np

Kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
Ky = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]])

def convolve(image, kernel):
    """Apply a 2D convolution between an image and a kernel."""
    k = kernel.shape[0] // 2
    padded = np.pad(image, ((k, k), (k, k)), mode='constant')
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+2*k+1, j:j+2*k+1]
            output[i, j] = np.sum(region * kernel)

    return output

def sobel_edge_detection(image_path):
    """
    Perform Sobel edge detection on an image.
    Returns the original image, combined Sobel edge image, X gradient, Y gradient, and their average.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    Gx = convolve(gray, Kx)
    Gy = convolve(gray, Ky)

    sobel_combined = np.sqrt(Gx**2 + Gy**2)
    sobel_combined = np.clip(sobel_combined, 0, 255).astype(np.uint8)

    abs_grad_x = np.abs(Gx).astype(np.uint8)
    abs_grad_y = np.abs(Gy).astype(np.uint8)

    grad_combined = (abs_grad_x * 0.5 + abs_grad_y * 0.5).astype(np.uint8)

    return image, sobel_combined, abs_grad_x, abs_grad_y, grad_combined
