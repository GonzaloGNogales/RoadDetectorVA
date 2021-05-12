import argparse
from MSERDetector.mser_detector import *
from DetectorUtilities.detector_visualizer import *
from DoubleMSERDetector.double_equalized_mser import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains and executes a given detector over a set of testing images')
    parser.add_argument('--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument('--train_path', default="", help='Select the training data dir')
    parser.add_argument('--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    detector = None
    # Create the detector
    if vars(args)['detector'] == 'mser':
        detector = MSER_Detector()
    elif vars(args)['detector'] == 'double_equalized_mser':
        detector = Double_Equalized_MSER_Detector()

    if detector is not None:
        # Load training data
        detector.preprocess_data(vars(args)['train_path'])

        # Training
        training_results = detector.fit()

        # DEBUG OR VISUALIZE TRAINING RESULTS
        visualizer = Detector_Visualizer(training_results)
        visualizer.setup_directory()
        visualizer.save_to_dir()

        # NOT IMPLEMENTED YET
        # Load testing data
        # detector.preprocess_data()

        # Evaluate sign detections -> output/gt.txt = [detector.predict(test_image) for test_image in test_images]
        # detector.predict()
    else:
        print("Detector is not defined")


    # cv2.imshow('Forbid Signal Resulting Mask', detector.forbid_mask)
    # cv2.imshow('Warning Signal Resulting Mask', detector.warning_mask)
    # cv2.imshow('Stop Signal Resulting Mask', detector.stop_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
