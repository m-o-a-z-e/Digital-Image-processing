"""
color_manipulation.py

This module provides functions for manipulating the color channels of images using OpenCV.
It allows you to increase or decrease the intensity of red, green, or blue channels,
swap color channels, and eliminate specific color channels from an image.
made by [Abdelrahman Salah]
"""

import cv2 as cv
import numpy as np

# Increase or decrease the intensity of the red channel in the image
def change_red(image_path, brightness):
    image = cv.imread(image_path)
    image[:, :, 2] = cv.add(image[:, :, 2], brightness)
    return image

# Increase or decrease the intensity of the green channel in the image
def change_green(image_path, brightness):
    image = cv.imread(image_path)
    image[:, :, 1] = cv.add(image[:, :, 1], brightness)
    return image

# Increase or decrease the intensity of the blue channel in the image
def change_blue(image_path, brightness):
    image = cv.imread(image_path)
    image[:, :, 0] = cv.add(image[:, :, 0], brightness)
    return image

# Swap the red and green channels in the image
def swap_red_green(image_path):
    image = cv.imread(image_path)
    image[:, :, [0, 1, 2]] = image[:, :, [0, 2, 1]]
    return image

# Swap the red and blue channels in the image
def swap_red_blue(image_path):
    image = cv.imread(image_path)
    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
    return image

# Swap the green and blue channels in the image
def swap_green_blue(image_path):
    image = cv.imread(image_path)
    image[:, :, [0, 1, 2]] = image[:, :, [1, 0, 2]]
    return image

# Eliminate the red channel from the image
def eliminate_red(image_path):
    image = cv.imread(image_path)
    image[:, :, 2] = 0
    return image

# Eliminate the green channel from the image
def eliminate_green(image_path):
    image = cv.imread(image_path)
    image[:, :, 1] = 0
    return image

# Eliminate the blue channel from the image
def eliminate_blue(image_path):
    image = cv.imread(image_path)
    image[:, :, 0] = 0
    return image