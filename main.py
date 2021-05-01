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

class MSER_Detector:
    def __init__(self, delta=10, max_variation=0.25, max_area=1000, min_area=50):
        self.original_images = list()  # Original lists vector for saving the original data (color detection)
        self.greyscale_images = list()  # Input list containing Greyscale images to feed MSER Detector (localization with MSER)

        # Initialize Maximally Stable Extremal Regions (MSER)
        self.mser = cv2.MSER_create(_delta=delta, _max_variation=max_variation, _max_area=max_area, _min_area=min_area)

    def preprocess_data(self, directory='train_10_ejemplos/'):
        # Read the input training images in color
        for actual in os.listdir(directory):
            if actual != 'gt.txt':  # Exclude txt containing signal information (location and class)
                self.original_images.append(cv2.imread(directory + actual))

        # 1 - Create another list with the greyscale recolored images
        for idx in range(len(self.original_images)):
            self.greyscale_images.append(cv2.cvtColor(self.original_images[idx], cv2.COLOR_BGR2GRAY))

    # DETECTOR TRAINING
    def fit(self, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type=cv2.THRESH_BINARY, block_size=11, c=12):
        batch_size = len(self.greyscale_images)
        mser_outputs = list()  # Output list containing MSER detected regions
        training_output = {}

        for i in range(batch_size):
            # 2 - Filter with adaptive threshold for increasing the contrast of the points of interest
            self.greyscale_images[i] = cv2.adaptiveThreshold(self.greyscale_images[i], max_value, adaptive_method, threshold_type, block_size, c)
            mser_outputs.append(np.zeros((self.original_images[i].shape[0], self.original_images[i].shape[1], 3), dtype=np.uint8))

            # Detect polygons (regions) from the image
            polygons = self.mser.detectRegions(self.greyscale_images[i])

            # Color output ***********************************************************************************************************************
            for polygon in polygons[0]:
                color_RGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
                color_RGB = tuple(int(color * 255) for color in color_RGB)
                mser_outputs[i] = cv2.fillPoly(mser_outputs[i], [polygon], color_RGB)
            # ************************************************************************************************************************************

            # Color rectangles
            candidate_regions = list()
            filtered_detected_regions = list()
            masks = list()
            for polygon in polygons[0]:
                x, y, w, h = cv2.boundingRect(polygon)
                if abs(1 - w / h) <= 0.2:
                    color_RGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
                    color_RGB = tuple(int(color * 255) for color in color_RGB)
                    x -= 5
                    y -= 5
                    w += 10
                    h += 10
                    candidate_regions.append(polygon)  # Save polygon into candidate regions
                    crop_img = self.original_images[i][y:y + h, x:x + w]
                    h, w, _ = crop_img.shape

                    if h <= 0 or w <= 0:  # Control test for not getting inconsistent dimensions
                        continue

                    # Red levels in HSV approximation
                    low_red_1 = np.array([0, 100, 20])
                    high_red_1 = np.array([8, 255, 255])
                    low_red_2 = np.array([175, 100, 20])
                    high_red_2 = np.array([179, 255, 255])

                    # Red mask creation for HSV color thresholding
                    crop_img_HSV = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                    red_mask_1 = cv2.inRange(crop_img_HSV, low_red_1, high_red_1)
                    red_mask_2 = cv2.inRange(crop_img_HSV, low_red_2, high_red_2)
                    red_mask = cv2.add(red_mask_1, red_mask_2)

                    # Establish a threshold for discriminating the probable signal red masks from the others
                    red_mask_mean = np.mean(red_mask)
                    red_mask = cv2.resize(red_mask, (25, 25))
                    if 25 < red_mask_mean < 50:
                        filtered_detected_regions.append(crop_img)
                        masks.append(red_mask)

            # *********************************************** DEBUG ********************************************
            cv2.rectangle(self.original_images[i], (x, y), (x + w, y + h), color_RGB, 2)
            # we only want the regions not painting the rectangles
            # **************************************************************************************************

            training_output[i] = (masks, filtered_detected_regions, mser_outputs)

        return training_output

    # DETECTOR TESTING
    def predict(self):
        # 3 - Not implemented yet
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains and executes a given detector over a set of testing images')
    parser.add_argument('--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument('--train_path', default="", help='Select the training data dir')
    parser.add_argument('--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()
    print(vars(args)['detector'])

    # Create the detector
    detector = MSER_Detector()

    # Load training data
    detector.preprocess_data()

    # Training
    training_results = detector.fit()

    # Show training results on screen
    for act_result in range(len(training_results)):
        masks, regions, mser = training_results[act_result]

        for m in range(len(masks)):
            masks[m] = cv2.resize(masks[m], (500, 500))
            regions[m] = cv2.resize(regions[m], (500, 500))
            cv2.imshow('Mask Region Image ' + str(m) + ':', regions[m])
            cv2.imshow('Red Mask Image ' + str(m) + ':', masks[m])

        for d in range(len(mser)):
            mser[d] = cv2.resize(mser[d], (500, 500))
            detector.original_images[d] = cv2.resize(detector.original_images[d], (500, 500))  # DEBUG ONLY
            cv2.imshow('MSER Regions Image ' + str(d) + ':', mser[d])
            cv2.imshow('Original Rect Bounded Image ' + str(d) + ':', detector.original_images[d])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # NOT IMPLEMENTED YET
    # Load testing data
    # detector.preprocess_data()

    # Evaluate sign detections -> output/gt.txt = [detector.predict(test_image) for test_image in test_images]
    # detector.predict()

