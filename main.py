import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import random
from colorsys import hsv_to_rgb

# El objetivo de la práctica 1 a entregar es desarrollar un detector muy básico de señales de tráfico para
# los tres tipos principales: prohibición (con fondo blanco), peligro y stop.

# Tipos de señales:
# Prohibición: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
# Peligro: [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# Stop: [14]

# Read the input training images in color
images = list()
for i in os.listdir('train_10_ejemplos'):
    if i != 'gt.txt':
        images.append(cv2.imread('train_10_ejemplos/' + i))
N = len(images)

# 1 - Create another list with the greyscale recolored images
greyscale_images = list()
for i in range(N):
    greyscale_images.append(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY))

# MSER OPERATIONS

# Initialize Maximally Stable Extremal Regions (MSER) and output matrix list
outputs = list()
mser = cv2.MSER_create(_delta=10, _max_variation=0.25, _max_area=1000, _min_area=50)

masks = list()
originals = list()
for i in range(N):
    # 2 - Filter with adaptive threshold for increasing the contrast of the points of interest
    greyscale_images[i] = cv2.adaptiveThreshold(greyscale_images[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    outputs.append(np.zeros((images[i].shape[0], images[i].shape[1], 3), dtype=np.uint8))

    # Detect polygons (regions) from the image
    polygons = mser.detectRegions(greyscale_images[i])

    # Color output
    for polygon in polygons[0]:
        colorRGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
        colorRGB = tuple(int(color*255) for color in colorRGB)
        outputs[i] = cv2.fillPoly(outputs[i], [polygon], colorRGB)

    # Color rectangles
    candidate_regions = list()
    for polygon in polygons[0]:
        x, y, w, h = cv2.boundingRect(polygon)
        if abs(1 - w / h) <= 0.2:
            colorRGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
            colorRGB = tuple(int(color*255) for color in colorRGB)
            x -= 5
            y -= 5
            w += 10
            h += 10
            # cv2.rectangle(images[i], (x, y), (x + w, y + h), colorRGB, 2)  # we only want the regions not painting the rectangles
            candidate_regions.append(polygon)  # Save polygon into candidate regions
            crop_img = images[i][y:y+h, x:x+w]
            h, w, _ = crop_img.shape

            if h <= 0 or w <= 0:
                continue

            low_red_1 = np.array([0, 100, 20])
            high_red_1 = np.array([8, 255, 255])
            low_red_2 = np.array([175, 100, 20])
            high_red_2 = np.array([179, 255, 255])

            crop_img_HSV = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            red_mask_1 = cv2.inRange(crop_img_HSV, low_red_1, high_red_1)
            red_mask_2 = cv2.inRange(crop_img_HSV, low_red_2, high_red_2)
            red_mask = cv2.add(red_mask_1, red_mask_2)

            # Establish a threshold for discriminating the probable signal red masks from the others
            red_mask_mean = np.mean(red_mask)
            red_mask = cv2.resize(red_mask, (25, 25))
            if 25 < red_mask_mean < 50:
                originals.append(crop_img)
                masks.append(red_mask)


for m in range(len(masks)):
    masks[m] = cv2.resize(masks[m], (500, 500))
    originals[m] = cv2.resize(originals[m], (500, 500))

    cv2.imshow('Red Mask Image ' + str(m) + ':', masks[m])
    cv2.imshow('Original Image ' + str(m) + ':', originals[m])

# # Show output
# for i in range(N):
#     cv2.imshow('Rect Detected Image ' + str(i) + ':', images[i])
#     cv2.imshow('Greyscaled Image ' + str(i) + ':', greyscale_images[i])
#     cv2.imshow('MSER ' + str(i) + ':', outputs[i])
#     break
#
cv2.waitKey(0)
cv2.destroyAllWindows()

# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description='Trains and executes a given detector over a set of testing images')
#     parser.add_argument('--detector', type=str, nargs="?", default="", help='Detector string name')
#     parser.add_argument('--train_path', default="", help='Select the training data dir')
#     parser.add_argument('--test_path', default="", help='Select the testing data dir')
#
#     args = parser.parse_args()
#
#
#
#     # Load training data
#
#     # Create the detector
#
#     # Load testing data
#
#     # Evaluate sign detections
#

