import os
import cv2
from PIL import Image
import shutil


class Detector_Visualizer:
    def __init__(self, data_from_detector, folder_name='DetectionResults/'):
        # Tuple of data containing 3 image vectors (masks, filtered regions and mser regions)
        self.data = data_from_detector
        # Directory name for saving the results
        self.dir = folder_name

    def setup_directory(self):
        # Check if results folder already exists and clear it, if not create it
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        else:
            shutil.rmtree(self.dir)
            os.mkdir(self.dir)

    def save_to_dir(self):
        # Show training results on screen & save the same results in a folder self.dir
        for key in self.data:
            masks, regions, mser, rect_detections, equalization_result = self.data[key]

            characteristics = list()  # List of detected characteristics for storing the elements of a single image detection
            for m in range(len(masks)):
                # Resize the masks and regions detected on an image and save them in the characteristics vector for displaying later
                mask, _ = masks[m]
                mask = cv2.resize(mask, (500, 500))
                regions[m] = cv2.resize(regions[m], (500, 500))
                regions[m] = regions[m][:, :, ::-1]  # We have to shuffle the channels for getting RGB display of images
                characteristics.append(Image.fromarray(mask))
                characteristics.append(Image.fromarray(regions[m]))

            mser[key] = cv2.resize(mser[key], (500, 500))
            rect_detections = cv2.resize(rect_detections, (500, 500))  # DEBUG ONLY
            rect_detections = rect_detections[:, :, ::-1]
            equalization_result = cv2.resize(equalization_result, (500, 500))
            characteristics.append(Image.fromarray(mser[key], 'RGB'))
            characteristics.append(Image.fromarray(rect_detections, 'RGB'))
            characteristics.append(Image.fromarray(equalization_result))

            # Create a new image containing the characteristics detected by the detector and the original image
            # with the rectangles displayed for adding more legibility and enabling better debug practices
            characteristics_total_width = 500 * len(characteristics)
            characteristics_max_height = 500
            new_characteristics_img = Image.new('RGB', (characteristics_total_width, characteristics_max_height))
            xc_offset = 0
            for c in characteristics:
                new_characteristics_img.paste(c, (xc_offset, 0))
                xc_offset += c.size[0]

            # Finally save the created image with an information flag [DETECTED] if our detector classified something
            # as a signal or [NO DETECTIONS] for clarifying the absence of results (or detected areas)
            if len(characteristics) > 3:
                new_characteristics_img.save('DetectionResults/[DETECTED] Image Results - ' + str(key)[0:5] + '.jpg')
            else:
                new_characteristics_img.save(
                    'DetectionResults/[NO DETECTIONS] Image Results - ' + str(key)[0:5] + '.jpg')
