import os
import cv2
import numpy as np
from random import random
from colorsys import hsv_to_rgb


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
            regions, _ = self.mser.detectRegions(self.greyscale_images[i])

            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            cv2.polylines(self.original_images[i], hulls, 1, (0, 255, 0))
            cv2.imshow('result', self.original_images[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Color MSER output **************************************************************************************
            for polygon in regions:  # polygons[0]:
                color_RGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
                color_RGB = tuple(int(color * 255) for color in color_RGB)
                mser_outputs[i] = cv2.fillPoly(mser_outputs[i], [polygon], color_RGB)
            # ********************************************************************************************************

            # Color rectangles and Mask extraction
            candidate_regions = list()
            filtered_detected_regions = list()
            masks = list()
            original_image = np.copy(self.original_images[i])
            for polygon in regions:  # polygons[0]:
                x, y, w, h = cv2.boundingRect(polygon)
                if abs(1 - w / h) <= 0.2:
                    color_RGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
                    color_RGB = tuple(int(color * 255) for color in color_RGB)
                    x -= 5
                    y -= 5
                    w += 10
                    h += 10
                    candidate_regions.append(polygon)  # Save polygon into candidate regions
                    crop_img = original_image[y:y + h, x:x + w]  # crop_img = self.original_images[i][y:y + h, x:x + w]
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
                    # we only want the regions not painting the rectangles for the final version
                    # **************************************************************************************************

            training_output[i] = (masks, filtered_detected_regions, mser_outputs, self.original_images[i])

        return training_output

    # DETECTOR TESTING
    def predict(self):
        # 3 - Not implemented yet
        return
