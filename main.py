import argparse
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

in_img = cv2.imread('train/00000.ppm')
print(in_img.shape)

# 1 - Convert the image color to grayscale
img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)

# MSER OPERATIONS
# 2 - Filter with some operations for managing contrast
# img = cv2.equalizeHist(img)  -> try equalization
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

# Initialize Maximally Stable Extremal Regions (MSER) and output matrix
output = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
mser = cv2.MSER_create(_delta=10, _max_variation=0.25, _max_area=1000, _min_area=50)

# Detect polygons (regions) from the image
polygons = mser.detectRegions(img)

# Color output
# for polygon in polygons[0]:
#     colorRGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
#     colorRGB = tuple(int(color*255) for color in colorRGB)
#     output = cv2.fillPoly(output, [polygon], colorRGB)

# Color rectangles
for polygon in polygons[0]:
    x, y, w, h = cv2.boundingRect(polygon)
    colorRGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
    colorRGB = tuple(int(color*255) for color in colorRGB)
    cv2.rectangle(in_img, (x, y), (x + w, y + h), colorRGB, 2)

# Show output
cv2.imshow('Rect Detected Image: ', in_img)
cv2.imshow('Gray Image: ', img)
cv2.imshow('MSER: ', output)

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

