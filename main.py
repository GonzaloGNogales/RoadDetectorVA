import argparse
from MSERDetector.mser_detector import *
from MSERDetector.detector_visualizer import *

# Signal types:
# Forbid: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
# Danger: [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# Stop: [14]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains and executes a given detector over a set of testing images')
    parser.add_argument('--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument('--train_path', default="", help='Select the training data dir')
    parser.add_argument('--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # Create the detector
    if vars(args)['detector'] == 'mser':
        detector = MSER_Detector()

        # Load training data
        detector.preprocess_data(vars(args)['train_path'])

        # Training
        training_results = detector.fit()

        # DEBUG OR VISUALIZE TRAINING RESULTS
        visualizer = Detector_Visualizer(training_results)
        visualizer.setup_directory()
        visualizer.show_and_save()

        # NOT IMPLEMENTED YET
        # Load testing data
        # detector.preprocess_data()

        # Evaluate sign detections -> output/gt.txt = [detector.predict(test_image) for test_image in test_images]
        # detector.predict()
