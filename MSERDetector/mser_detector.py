import os
import cv2
import numpy as np
from random import random
from colorsys import hsv_to_rgb
from DetectorUtilities.region import *


# Signal types:
# Forbid: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
# Warning: [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# Stop: [14]
# Ceda: [13]

class MSER_Detector:
    def __init__(self, delta=3, max_variation=0.2, max_area=2000, min_area=50):
        self.original_images = {}  # Original map for saving the original data (color detection)
        self.greyscale_images = {}  # Map containing Greyscale images to feed MSER Detector (localization with MSER)
        self.ground_truth = {}  # Map containing the regions of the present signals in the training images

        # Initialize Maximally Stable Extremal Regions (MSER)
        self.mser = cv2.MSER_create(_delta=delta, _max_variation=max_variation, _max_area=max_area, _min_area=min_area)

        # Initialize the resulting masks of the detector training
        self.forbid_mask = None
        self.warning_mask = None
        self.stop_mask = None

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
                            region = Region(components[0], components[1], components[2], components[3], components[4],
                                            components[5])
                            regions_set = self.ground_truth.get(region.file_name)
                            if regions_set is None:  # Check if the regions set of this image is empty yet
                                self.ground_truth[region.file_name] = set()
                            self.ground_truth[region.file_name].add(region)

        # 1 - Create another list with the greyscale recolored images
        for key in self.original_images:
            self.greyscale_images[key] = (cv2.cvtColor(self.original_images[key], cv2.COLOR_BGR2GRAY))

    # DETECTOR TRAINING
    def fit(self):
        forbid_set = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
        warning_set = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
        stop_set = {14}
        yield_set = {13}
        mser_outputs = {}  # Output map containing MSER detected regions
        training_output = {}  # Map for storing the outputs of the training phase, for visualization purposes
        forbid_masks_list = list()
        warning_masks_list = list()
        stop_masks_list = list()

        for act_img in self.greyscale_images:
            # 2 - Filter equalizing the histogram for increasing the contrast of the points of interest
            mser_outputs[act_img] = (
                np.zeros((self.original_images[act_img].shape[0], self.original_images[act_img].shape[1], 3),
                         dtype=np.uint8))

            # Detect polygons (regions) from the image
            regions, _ = self.mser.detectRegions(self.greyscale_images[act_img])

            # Color rectangles and Mask extraction
            filtered_detected_regions = list()
            masks = list()
            original_image = np.copy(self.original_images[act_img])
            best_regions = {}
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                if abs(1 - (w / h)) <= 0.4:  # Filter detected regions with an aspect ratio very different from a square
                    # Adjust the width and height to obtain a perfect square for the region
                    x = max(x - 5, 0)
                    y = max(y - 5, 0)
                    w += 10
                    h += 10
                    if w > h:
                        h = w
                    elif h > w:
                        w = h

                    reg = Region('', x, y, x + w, y + h)  # Instantiate an object to store the actual candidate region

                    # Search the candidate region through the possible known ground-truth regions of the actual image
                    if self.ground_truth.get(act_img) is not None:
                        for r in self.ground_truth[act_img]:
                            if reg == r:
                                """
                                Thanks to the code structure that we made we can check this with
                                the equals function of Region objects that redefines the == operator.
                                In this definition of equals we consider the error that can exist when
                                checking if a region is the same as the one in the ground-truth of an image.                          
                                The scope of this is to reduce time complexity and to improve efficiency.
                                """
                                # Classify the region type when found in the actual image ground-truth
                                reg.type = int(r.type)
                                error = abs(reg.x1 - int(r.x1)) + abs(reg.y1 - int(r.y1)) + \
                                        abs(reg.x2 - int(r.x2)) + abs(reg.y2 - int(r.y2))
                                if best_regions.get(reg.type) is None:
                                    best_regions[reg.type] = (reg, error)
                                else:
                                    _, last_error = best_regions[reg.type]
                                    if last_error > error:
                                        best_regions[reg.type] = (reg, error)

            for best_region, _ in best_regions.values():

                crop_region = original_image[int(best_region.y1):int(best_region.y2),
                              int(best_region.x1):int(best_region.x2)]  # Take the region from the original image

                # Red levels in HSV approximation
                low_red_mask_1 = np.array([0, 120, 20])
                high_red_mask_1 = np.array([8, 240, 255])
                low_red_mask_2 = np.array([175, 120, 20])
                high_red_mask_2 = np.array([179, 240, 255])

                # Orange levels in HSV approximation
                low_orange_mask = np.array([5, 100, 80])
                high_orange_mask = np.array([15, 140, 140])

                # Dark Red levels in HSV approximation
                low_darkred_mask = np.array([170, 25, 35])
                # high_darkred_mask = np.array([179, 70, 100])
                high_darkred_mask = np.array([179, 170, 180])

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
                if 10 < red_mask_mean < 70 or 20 < orange_mask_mean < 70 or 15 < darkred_mask_mean < 70:
                    filtered_detected_regions.append(crop_region)
                    if red_mask_mean > orange_mask_mean and red_mask_mean > darkred_mask_mean:
                        masks.append((red_mask, best_region))
                    elif orange_mask_mean > red_mask_mean and orange_mask_mean > darkred_mask_mean:
                        masks.append((orange_mask, best_region))
                    else:
                        masks.append((darkred_mask, best_region))

            # Montar las listas de mascaras clasificandolas de forma eficiente usando sets
            for mask, mask_region in masks:
                if mask_region.type in forbid_set:
                    forbid_masks_list.append(mask)
                elif mask_region.type in stop_set:
                    stop_masks_list.append(mask)
                elif mask_region.type in yield_set:
                    mask = cv2.rotate(mask, cv2.ROTATE_180)
                    warning_masks_list.append(mask)
                elif mask_region.type in warning_set:
                    warning_masks_list.append(mask)

            training_output[act_img] = (masks,
                                        filtered_detected_regions,
                                        mser_outputs,
                                        self.original_images[act_img],
                                        self.greyscale_images[act_img])

        # Aqui tenemos que sacar la mascara media de cada tipo de entre todas las que tenemos
        # Iterar las listas de mascaras y cuando tengamos la mascara media la guardamos en los atributos del detector
        total_forbid = len(forbid_masks_list)
        total_warning = len(warning_masks_list)
        total_stop = len(stop_masks_list)

        if total_forbid != 0:
            sum_forbid = forbid_masks_list[0]
            for f in range(1, total_forbid):
                cv2.add(sum_forbid, forbid_masks_list[f])
                # sum_forbid += forbid_masks_list[f]
            self.forbid_mask = sum_forbid / total_forbid
        if total_warning != 0:
            sum_warning = warning_masks_list[0]
            for w in range(1, total_warning):
                cv2.add(sum_warning, warning_masks_list[w])
                # sum_warning += warning_masks_list[w]
            self.warning_mask = sum_warning / total_warning
        if total_stop != 0:
            sum_stop = stop_masks_list[0]
            for s in range(1, total_stop):
                cv2.add(sum_stop, stop_masks_list[s])
                # sum_stop += stop_masks_list[s]
            self.stop_mask = sum_stop / total_stop

        return training_output

    # DETECTOR TESTING
    def predict(self):
        # 3 - Not implemented yet

        return
