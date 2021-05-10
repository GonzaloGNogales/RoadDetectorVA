import os
import cv2
import numpy as np
from random import random
from colorsys import hsv_to_rgb
from DetectorUtilities.region import *


class MSER_Detector:
    def __init__(self, delta=3, max_variation=0.2, max_area=2000, min_area=50):
        self.original_images = {}   # Original map for saving the original data (color detection)
        self.greyscale_images = {}  # Map containing Greyscale images to feed MSER Detector (localization with MSER)
        self.ground_truth = {}      # Map containing the regions of the present signals in the training images

        # Initialize Maximally Stable Extremal Regions (MSER)
        self.mser = cv2.MSER_create(_delta=delta, _max_variation=max_variation, _max_area=max_area, _min_area=min_area)

        # self.forbid_mask
        # self.warning_mask
        # self.stop_mask

    def preprocess_data(self, directory='train_10_ejemplos/'):
        # Read the input training images in color
        for actual in os.listdir(directory):
            if actual != 'gt.txt':  # Exclude txt containing signal information (location and class)
                self.original_images[actual] = (cv2.imread(directory + actual))
            else:  # Initialize regions for training through the ground-truth file
                with open(directory + actual) as gt:
                    lines = gt.readlines()
                    for line in lines:
                        # Extract components of the ground-truth line for creating a region object
                        components = line.split(";")
                        if len(components) == 6:  # Check if there are enough components to instantiate a region
                            region = Region(components[0], components[1], components[2], components[3], components[4], components[5])
                            regions_set = self.ground_truth.get(region.file_name)
                            if regions_set is None:  # Check if the regions set of this image is empty yet
                                self.ground_truth[region.file_name] = set()
                            self.ground_truth[region.file_name].add(region)

        # 1 - Create another list with the greyscale recolored images
        for key in self.original_images:
            self.greyscale_images[key] = (cv2.cvtColor(self.original_images[key], cv2.COLOR_BGR2GRAY))

    # DETECTOR TRAINING
    def fit(self):
        mser_outputs = {}      # Output map containing MSER detected regions
        training_output = {}   # Map for storing the outputs of the training phase, for visualization purposes

        for key in self.greyscale_images:
            # 2 - Filter equalizing the histogram for increasing the contrast of the points of interest
            equalized_greyscale_image = cv2.equalizeHist(self.greyscale_images[key])
            mser_outputs[key] = (np.zeros((self.original_images[key].shape[0], self.original_images[key].shape[1], 3), dtype=np.uint8))

            # Detect polygons (regions) from the image
            regions_non_equalized, _ = self.mser.detectRegions(self.greyscale_images[key])
            regions_equalized, _ = self.mser.detectRegions(equalized_greyscale_image)
            regions = regions_non_equalized + regions_equalized

            # Color MSER output **************************************************************************************
            for region in regions:
                color_RGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
                color_RGB = tuple(int(color * 255) for color in color_RGB)
                mser_outputs[key] = cv2.fillPoly(mser_outputs[key], [region], color_RGB)
            # ********************************************************************************************************

            # Color rectangles and Mask extraction
            filtered_detected_regions = list()
            masks = list()
            original_image = np.copy(self.original_images[key])
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                if abs(1 - w / h) <= 0.8:  # Filter detected regions with an aspect ratio very different from a square
                    color_RGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
                    color_RGB = tuple(int(color * 255) for color in color_RGB)

                    # Adjust the width and height to obtain a perfect square for the region
                    x = max(x - 5, 0)
                    y = max(y - 5, 0)
                    w += 10
                    h += 10
                    if w > h:
                        h = w
                    elif h > w:
                        w = h

                    reg = Region('', x, y, x + w, y + h)  # Instantiate an object to store the actual bounding region

                    # Search the candidate region through the possible known ground-truth regions of the actual image
                    for r in self.ground_truth[key]:
                        if reg == r:
                            """
                            Thanks to the code structure that we made we can check this with
                            the equals function of Region objects that redefines the == operator.
                            In this definition of equals we consider the error that can exist when
                            checking if a region is the same as the one in the ground-truth of an image.                          
                            The scope of this is to reduce time complexity and to improve efficiency.
                            """

                            crop_region = original_image[y:y + h, x:x + w]  # Take the region from the original image

                            # Red levels in HSV approximation
                            low_red_mask_1 = np.array([0, 100, 20])
                            high_red_mask_1 = np.array([8, 255, 255])  # Review!!!
                            low_red_mask_2 = np.array([175, 100, 20])
                            high_red_mask_2 = np.array([179, 255, 255])

                            # Orange levels in HSV approximation
                            low_orange_mask = np.array([5, 100, 20])
                            high_orange_mask = np.array([15, 255, 255])

                            # Dark Red levels in HSV approximation
                            low_darkred_mask = np.array([170, 20, 30])
                            high_darkred_mask = np.array([179, 70, 100])

                            # Red mask, Orange mask and Dark red mask creation for HSV color thresholding
                            crop_img_HSV = cv2.cvtColor(crop_region, cv2.COLOR_BGR2HSV)
                            red_mask_1 = cv2.inRange(crop_img_HSV, low_red_mask_1, high_red_mask_1)
                            red_mask_2 = cv2.inRange(crop_img_HSV, low_red_mask_2, high_red_mask_2)
                            red_mask = cv2.add(red_mask_1, red_mask_2)
                            orange_mask = cv2.inRange(crop_img_HSV, low_orange_mask, high_orange_mask)
                            darkred_mask = cv2.inRange(crop_img_HSV, low_darkred_mask, high_darkred_mask)

                            # Establish a threshold for discriminating the probable signal
                            # red, orange and dark red masks from the others
                            darkred_mask = cv2.resize(darkred_mask, (25, 25))
                            orange_mask = cv2.resize(orange_mask, (25, 25))
                            red_mask = cv2.resize(red_mask, (25, 25))

                            # Calculate the mean of the masks for filtering the candidate regions:
                            # if a regions has lower than 10% of the mask color present or more than 80%
                            # discards it and the algorithm continues with the next candidate region
                            red_mask_mean = red_mask.mean()
                            orange_mask_mean = orange_mask.mean()
                            darkred_mask_mean = darkred_mask.mean()
                            if 10 < red_mask_mean < 80 or 20 < orange_mask_mean < 80 or 15 < darkred_mask_mean < 80:
                                filtered_detected_regions.append(crop_region)
                                if red_mask_mean > orange_mask_mean and red_mask_mean > darkred_mask_mean:
                                    masks.append(red_mask)
                                elif orange_mask_mean > red_mask_mean and orange_mask_mean > darkred_mask_mean:
                                    masks.append(orange_mask)
                                else:
                                    masks.append(darkred_mask)

                            # *********************************************** DEBUG ************************************
                            cv2.rectangle(self.original_images[key], (x, y), (x + w, y + h), color_RGB, 2)
                            # we only want the regions for debugging, not painting the rectangles for the final version
                            # ******************************************************************************************

            training_output[key] = (masks,
                                    filtered_detected_regions,
                                    mser_outputs,
                                    self.original_images[key],
                                    self.greyscale_images[key])

        return training_output

    # DETECTOR TESTING
    def predict(self):
        # 3 - Not implemented yet
        return
