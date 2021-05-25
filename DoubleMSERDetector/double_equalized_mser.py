import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
from DetectorUtilities.region import *
from DetectorUtilities.progress_bar import *


# Helper function for saving our training accuracy function to a file for visualization purposes
def save_training_metrics(x, y, x_prime, y_prime):
    train_total = np.array(y).sum()
    gt_total = np.array(y_prime).sum()
    train_acc = (train_total / gt_total) * 100
    train_acc = round(train_acc, 4)
    accuracy_comparison = plt.figure(figsize=(15, 5))
    plt.plot(x, y, '--c', label='Training Accuracy')
    plt.plot(x_prime, y_prime, 'orange', label='Target Accuracy')
    plt.title('Accuracy => ' + str(train_acc) + '%')
    plt.xlabel("Training Image Index")
    plt.ylabel("Signal Detections")
    plt.legend()
    accuracy_comparison.savefig('double_mser_training_accuracy.png')
    print("Training metrics saved on double_mser_training_accuracy.png")


class Double_Equalized_MSER_Detector:
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
        # Pixel proportion of the masks to calculate the score in an easier way
        self.forbid_pixels_proportion = 0
        self.warning_pixels_proportion = 0
        self.stop_pixels_proportion = 0

    def preprocess_data(self, directory, train):
        # Clear the original and the greyscale images lists for reusing preprocessing test data with this function
        self.original_images.clear()
        self.greyscale_images.clear()
        # Read the input training images in color and show the progress in the terminal with a helper function
        total = len(os.listdir(directory))
        it = 0
        if train:
            progress_bar(it, total, prefix='Loading train data:', suffix='Complete', length=50)
        else:
            progress_bar(it, total, prefix='Loading test data: ', suffix='Complete', length=50)
        for actual in os.listdir(directory):
            if os.path.isfile(directory + actual):
                it += 1
                if train:
                    progress_bar(it, total, prefix='Loading train data:', suffix='Complete', length=50)
                else:
                    progress_bar(it, total, prefix='Loading test data: ', suffix='Complete', length=50)

                # Exclude txt containing signal information (location and class)
                if actual != 'gt.txt' and (actual.endswith('.ppm') or actual.endswith('.jpg')):
                    self.original_images[actual] = (cv2.imread(directory + actual))
                else:  # Initialize regions for training through the ground-truth file
                    with open(directory + actual) as gt:
                        lines = gt.readlines()
                        for line in lines:
                            # Extract components of the ground-truth line for creating a region object
                            components = line.split(";")
                            if len(components) == 6:  # Check if there are enough components to instantiate a region
                                region = Region(components[0], components[1], components[2], components[3],
                                                components[4],
                                                components[5])
                                regions_set = self.ground_truth.get(region.file_name)
                                if regions_set is None:  # Check if the regions set of this image is empty yet
                                    self.ground_truth[region.file_name] = set()
                                self.ground_truth[region.file_name].add(region)
            else:
                # If we are processing a directory with wrong format we undo everything we read and stop the algorithm
                if train:
                    print('The train directory is invalid and training data preprocess failed')
                else:
                    print('The test directory is invalid and testing data preprocess failed')
                self.original_images.clear()
                break

        # Create another list with the greyscale recolored images
        for ori_image in self.original_images:
            self.greyscale_images[ori_image] = (cv2.cvtColor(self.original_images[ori_image], cv2.COLOR_BGR2GRAY))

    # DETECTOR TRAINING
    def fit(self):
        # Sets for classifying the best found regions in O(1) complexity using (_region_ in _set_)
        forbid_set = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
        warning_set = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
        stop_set = {14}
        yield_set = {13}

        # Declare the masks lists for storing the already classified ones into forbid/warning/stop classes to calculate
        # their mean mask for the detector
        forbid_regions_list = list()
        warning_regions_list = list()
        stop_regions_list = list()

        # List to save the detected signals during training for metrics visualization
        our_accuracy = list()
        gt_accuracy = list()

        # Train with all the preprocessed images and show the progress in the terminal with the helper function
        it = 0
        total = len(self.greyscale_images)
        if total != 0:
            progress_bar(it, total, prefix='Training progress: ', suffix='Complete', length=50)
        for act_img in self.greyscale_images:
            it += 1
            progress_bar(it, total, prefix='Training progress: ', suffix='Complete', length=50)
            # Detect polygons (regions) from the train image using mser detect regions operation

            # Perform the maximal stable extreme regions detection (MSER) 2 times for getting the eq_reg & non_eq_reg
            eq_img = cv2.equalizeHist(self.greyscale_images[act_img])
            regions_non_eq, _ = self.mser.detectRegions(self.greyscale_images[act_img])
            regions_eq, _ = self.mser.detectRegions(eq_img)
            regions = regions_non_eq + regions_eq

            # Color rectangles and Mask extraction
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
                                # Calculate the offset error of the candidate region to filter regions that are not
                                # centered on the target, this is needed for reducing execution time and to save the
                                # best options, we will keep the found best option on the best regions map in O(1)
                                error = abs(reg.x1 - int(r.x1)) + abs(reg.y1 - int(r.y1)) + \
                                        abs(reg.x2 - int(r.x2)) + abs(reg.y2 - int(r.y2))
                                if best_regions.get(reg.type) is None:
                                    best_regions[reg.type] = (reg, error)
                                else:
                                    _, last_error = best_regions[reg.type]
                                    if last_error > error:
                                        best_regions[reg.type] = (reg, error)

            # We can save the correctly detected images for the metrics now and start the best regions classification
            our_accuracy.append(len(best_regions))
            if self.ground_truth.get(act_img) is not None:
                gt_accuracy.append(len(self.ground_truth[act_img]))
            else:
                gt_accuracy.append(0)
            for best_region, _ in best_regions.values():

                # Extract the region from the original image to process the mean
                crop_region = self.original_images[act_img][int(best_region.y1):int(best_region.y2),
                              int(best_region.x1):int(best_region.x2)]

                crop_region = cv2.resize(crop_region, (25, 25))  # Normalize all the regions to 25x25 pixels

                # Build mask lists classifying them using containing operation (in) on sets for O(1) complexity
                if best_region.type in forbid_set:
                    forbid_regions_list.append(crop_region)
                elif best_region.type in yield_set:  # Take yield signals and rotate them to act as warning regions
                    crop_region = cv2.rotate(crop_region, cv2.ROTATE_180)
                    warning_regions_list.append(crop_region)
                elif best_region.type in warning_set:
                    warning_regions_list.append(crop_region)
                elif best_region.type in stop_set:
                    stop_regions_list.append(crop_region)

        # Get length of mask lists for dividing in the mean calculation
        total_forbid = len(forbid_regions_list)
        total_warning = len(warning_regions_list)
        total_stop = len(stop_regions_list)
        forbid_mean_region = None
        warning_mean_region = None
        stop_mean_region = None

        # Mean region calculation block
        if total_forbid != 0:
            sum_forbid = np.float32(forbid_regions_list[0])
            for f in range(1, total_forbid):
                sum_forbid = cv2.add(sum_forbid, np.float32(forbid_regions_list[f]))
            forbid_mean_region = np.uint8(sum_forbid / total_forbid)
        if total_warning != 0:
            sum_warning = np.float32(warning_regions_list[0])
            for w in range(1, total_warning):
                sum_warning = cv2.add(sum_warning, np.float32(warning_regions_list[w]))
            warning_mean_region = np.uint8(sum_warning / total_warning)
        if total_stop != 0:
            sum_stop = np.float32(stop_regions_list[0])
            for s in range(1, total_stop):
                sum_stop = cv2.add(sum_stop, np.float32(stop_regions_list[s]))
            stop_mean_region = np.uint8(sum_stop / total_stop)
        if total != 0:
            print("Forbid|Warning|Stop regions mean calculation finished successfully!")

        # Now we calculate the mask of the mean forbid, warning and stop regions
        # Red levels in HSV approximation
        low_red_mask_1 = np.array([0, 65, 65])
        high_red_mask_1 = np.array([8, 255, 255])
        low_red_mask_2 = np.array([175, 75, 75])
        high_red_mask_2 = np.array([179, 255, 255])

        # Calculate red mask for forbid signals and their pixel proportion
        white_img = 255 * np.ones((25, 25), np.uint8)
        if forbid_mean_region is not None:
            forbid_img_HSV = cv2.cvtColor(forbid_mean_region, cv2.COLOR_BGR2HSV)
            f_red_mask_1 = cv2.inRange(forbid_img_HSV, low_red_mask_1, high_red_mask_1)
            f_red_mask_2 = cv2.inRange(forbid_img_HSV, low_red_mask_2, high_red_mask_2)
            self.forbid_mask = cv2.add(f_red_mask_1, f_red_mask_2)
            active_pixels = white_img * self.forbid_mask
            self.forbid_pixels_proportion = active_pixels.sum()

        # Calculate red mask for warning signals
        if warning_mean_region is not None:
            warning_img_HSV = cv2.cvtColor(warning_mean_region, cv2.COLOR_BGR2HSV)
            w_red_mask_1 = cv2.inRange(warning_img_HSV, low_red_mask_1, high_red_mask_1)
            w_red_mask_2 = cv2.inRange(warning_img_HSV, low_red_mask_2, high_red_mask_2)
            self.warning_mask = cv2.add(w_red_mask_1, w_red_mask_2)
            active_pixels = white_img * self.warning_mask
            self.warning_pixels_proportion = active_pixels.sum()

        # Calculate red mask for stop signals
        if stop_mean_region is not None:
            stop_img_HSV = cv2.cvtColor(stop_mean_region, cv2.COLOR_BGR2HSV)
            s_red_mask_1 = cv2.inRange(stop_img_HSV, low_red_mask_1, high_red_mask_1)
            s_red_mask_2 = cv2.inRange(stop_img_HSV, low_red_mask_2, high_red_mask_2)
            self.stop_mask = cv2.add(s_red_mask_1, s_red_mask_2)
            active_pixels = white_img * self.stop_mask
            self.stop_pixels_proportion = active_pixels.sum()

        # Return a status boolean to notify the predict function if everything finished correctly
        x_values = list()
        for i in range(len(self.greyscale_images)):
            x_values.append(i)
        save_training_metrics(x_values, our_accuracy, x_values, gt_accuracy)
        return self.forbid_mask is not None and self.warning_mask is not None and self.stop_mask is not None

    # DETECTOR TESTING
    def predict(self, train_status):
        if train_status:
            # Prepare the detection results directory
            # Check if results folder already exists and clear it, if not create it
            if os.path.isdir('resultado_imgs/'):
                shutil.rmtree('resultado_imgs/')
            os.mkdir('resultado_imgs/')

            if os.path.isfile('resultado_imgs/resultado.txt'):
                os.remove('resultado_imgs/resultado.txt')
            results = open('resultado_imgs/resultado.txt', 'w')

            # Test with all the preprocessed images and show the progress in the terminal with the helper function
            it = 0
            total = len(self.greyscale_images)
            if total != 0:
                progress_bar(it, total, prefix='Testing progress:  ', suffix='Complete', length=50)
            for act_img in self.greyscale_images:
                it += 1
                progress_bar(it, total, prefix='Testing progress:  ', suffix='Complete', length=50)

                # Equalize the histogram of the image to distribute the levels of color homogeneously
                equalized_greyscale_image = cv2.equalizeHist(self.greyscale_images[act_img])

                # Perform the maximal stable extreme regions detection (MSER) 2 times for getting the eq_reg & non_eq_reg
                regions_non_equalized, _ = self.mser.detectRegions(self.greyscale_images[act_img])
                regions_equalized, _ = self.mser.detectRegions(equalized_greyscale_image)
                regions = regions_non_equalized + regions_equalized

                detected_regions = set()  # Set to filter repetitions

                original_image = np.copy(self.original_images[act_img])
                for region in regions:
                    x, y, w, h = cv2.boundingRect(region)
                    if abs(1 - (
                            w / h)) <= 0.4:  # Filter detected regions with an aspect ratio very different from a square
                        # Adjust the width and height to obtain a perfect square for the region
                        x = max(x - 5, 0)
                        y = max(y - 5, 0)
                        w += 10
                        h += 10
                        if w > h:
                            h = w
                        elif h > w:
                            w = h

                        reg = Region('', x, y, x + w,
                                     y + h)  # Instantiate an object to store the actual candidate region

                        ###################################### REPETITIONS DELETION ###################################
                        # Check if this region contains some previously detected one
                        to_remove = list()
                        update = True
                        for r in detected_regions:
                            if reg.contains(r):
                                # Update
                                to_remove.append(r)
                            elif r.contains(reg):
                                update = False
                                break

                        # Add every region into the set to automatically filter similar regions
                        if update:
                            detected_regions.add(reg)

                        # Try to remove all contained regions
                        for r in to_remove:
                            detected_regions.remove(r)
                        ###############################################################################################

                for region in detected_regions:
                    # Extract the region from the original image to process the mask
                    crop_region = original_image[int(region.y1):int(region.y2),
                                  int(region.x1):int(region.x2)]

                    # Change each mser detected region to 25x25 to correlate with the training masks
                    crop_region = cv2.resize(crop_region, (25, 25))

                    # Extract the M red mask from the region to compare with the training ones
                    # Red levels in HSV approximation
                    low_red_mask_1 = np.array([0, 120, 20])
                    high_red_mask_1 = np.array([8, 240, 255])
                    low_red_mask_2 = np.array([175, 120, 20])
                    high_red_mask_2 = np.array([179, 240, 255])

                    # Orange levels in HSV approximation
                    low_orange_mask = np.array([5, 100, 20])
                    high_orange_mask = np.array([15, 140, 255])

                    # Dark Red levels in HSV approximation, this is needed for dark signals that are near black color
                    # low_darkpurple_mask = np.array([140, 25, 10])
                    # high_darkpurple_mask = np.array([145, 100, 80])
                    low_darkred_mask = np.array([172, 75, 20])
                    high_darkred_mask = np.array([179, 140, 100])
                    low_darkbrown_mask = np.array([8, 50, 20])
                    high_darkbrown_mask = np.array([18, 120, 100])

                    crop_img_HSV = cv2.cvtColor(crop_region, cv2.COLOR_BGR2HSV)  # Change to HSV
                    red_mask_1 = cv2.inRange(crop_img_HSV, low_red_mask_1, high_red_mask_1)
                    red_mask_2 = cv2.inRange(crop_img_HSV, low_red_mask_2, high_red_mask_2)
                    # dark_mask_1 = cv2.inRange(crop_img_HSV, low_darkpurple_mask, high_darkpurple_mask)
                    dark_mask_2 = cv2.inRange(crop_img_HSV, low_darkred_mask, high_darkred_mask)
                    dark_mask_3 = cv2.inRange(crop_img_HSV, low_darkbrown_mask, high_darkbrown_mask)
                    M_red = cv2.add(red_mask_1, red_mask_2)  # M is the red mask extracted from the detected region
                    M_orange = cv2.inRange(crop_img_HSV, low_orange_mask, high_orange_mask)
                    M_darkred = cv2.add(dark_mask_2, dark_mask_3)

                    # Filter what mask gets better definition for the doing the correlation
                    M_red_mean = M_red.mean()
                    M_orange_mean = M_orange.mean()
                    M_darkred_mean = M_darkred.mean()
                    if 10 <= M_red_mean <= 70 or 20 <= M_orange_mean <= 70 or 15 <= M_darkred_mean <= 70:
                        if M_darkred_mean >= M_orange_mean and M_darkred_mean >= M_red_mean:
                            M = M_darkred
                        elif M_orange_mean >= M_red_mean and M_orange_mean >= M_darkred_mean:
                            M = M_orange
                        else:
                            M = M_red

                        # Correlate M with each mask obtained by the training: Multiply element by element
                        forbid_correlated = M * self.forbid_mask
                        warning_correlated = M * self.warning_mask
                        stop_correlated = M * self.stop_mask

                        # Add all the matching points between them to obtain a correlation coefficient
                        forbid_corr_coefficient = forbid_correlated.sum()
                        warning_corr_coefficient = warning_correlated.sum()
                        stop_corr_coefficient = stop_correlated.sum()

                        # Score computation -> Number of coincident pixels / Total number of pixels * 100 (percent)
                        forbid_corr_score = int((forbid_corr_coefficient / self.forbid_pixels_proportion) * 100)
                        warning_corr_score = int((warning_corr_coefficient / self.warning_pixels_proportion) * 100)
                        stop_corr_score = int((stop_corr_coefficient / self.stop_pixels_proportion) * 100)

                        # This value can be adjusted for reducing noisy detections
                        # We decided to keep it lower for detecting the majority
                        # of signals even though it detects some noise too
                        no_signal_threshold = 10
                        no_stop_signal = 20  # As stop mask can detect lot of noise we put its threshold higher
                        if forbid_corr_score > no_signal_threshold \
                                or warning_corr_score > no_signal_threshold \
                                or stop_corr_score > no_stop_signal:

                            # Filter the scores to classify the regions reg.type = type
                            if stop_corr_score >= forbid_corr_score and stop_corr_score >= warning_corr_score:
                                region.type = 3
                                write_score = stop_corr_score
                            elif warning_corr_score >= forbid_corr_score and warning_corr_score >= stop_corr_score:
                                region.type = 2
                                write_score = warning_corr_score
                            else:
                                region.type = 1
                                write_score = forbid_corr_score

                            # Check if the region is already detected

                            # *************** Write the Ground Truth &  Draw Bounding Rectangles & Save ***********
                            results.write(act_img + ';' + str(region.x1) + ';' + str(region.y1)
                                          + ';' + str(region.x2) + ';' + str(region.y2)
                                          + ';' + str(region.type) + ';' + str(write_score) + '\n')

                            cv2.rectangle(self.original_images[act_img], (region.x1, region.y1), (region.x2, region.y2),
                                          (0, 0, 255), 2)
                            # *************************************************************************************

                # Save the detected regions image into the results directory
                cv2.imwrite(os.path.join('resultado_imgs/', act_img), self.original_images[act_img])

            print("Testing finished, you can find the results inside \"resultado_imgs\" directory :)")
            results.close()  # Results txt file communication closed

        else:
            print('Testing failed due to lack of some training masks')
