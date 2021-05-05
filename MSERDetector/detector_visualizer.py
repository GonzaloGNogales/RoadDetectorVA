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

    def show_and_save(self):
        # Show training results on screen & save the same results in a folder self.dir
        for act_result in range(len(self.data)):
            masks, regions, mser, rect_detections = self.data[act_result]

            characteristics = list()  # List of detected characteristics for storing the elements of a single image detection
            for m in range(len(masks)):
                # Resize the masks and regions detected on an image and save them in the characteristics vector for displaying later
                masks[m] = cv2.resize(masks[m], (500, 500))
                regions[m] = cv2.resize(regions[m], (500, 500))
                regions[m] = regions[m][:, :, ::-1]  # We have to shuffle the channels for getting RGB display of images
                characteristics.append(Image.fromarray(masks[m]))
                characteristics.append(Image.fromarray(regions[m]))

            mser[act_result] = cv2.resize(mser[act_result], (500, 500))
            rect_detections = cv2.resize(rect_detections, (500, 500))  # DEBUG ONLY
            rect_detections = rect_detections[:, :, ::-1]
            characteristics.append(Image.fromarray(mser[act_result], 'RGB'))
            characteristics.append(Image.fromarray(rect_detections, 'RGB'))

            # Create a new image containing the characteristics detected by the detector and the original image
            # with the rectangles displayed for adding more legibility and enabling better debug practices
            characteristics_total_width = 500 * len(characteristics)
            characteristics_max_height = 500
            new_characteristics_img = Image.new('RGB', (characteristics_total_width, characteristics_max_height))
            xc_offset = 0
            for c in characteristics:
                new_characteristics_img.paste(c, (xc_offset, 0))
                xc_offset += c.size[0]

            # Display the new image on screen before saving it into a directory
            # cv2.imshow('Image Results - ' + str(act_result), np.array(new_characteristics_img)[:, :, ::-1])  # <- COMMENT THIS LINE IF BATCH SIZE IS LARGE

            # Finally save the created image with an information flag [DETECTED] if our detector classified something
            # as a signal or [NO DETECTIONS] for clarifying the absence of results (or detected areas)
            if len(characteristics) > 2:
                new_characteristics_img.save('DetectionResults/[DETECTED] Image Results - ' + str(act_result) + '.jpg')
            else:
                new_characteristics_img.save(
                    'DetectionResults/[NO DETECTIONS] Image Results - ' + str(act_result) + '.jpg')

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
