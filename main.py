import argparse
from MSERDetector.mser_detector import *
from DoubleMSERDetector.double_equalized_mser import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains and executes a given detector over a set of testing images')
    parser.add_argument('--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument('--train_path', default="", help='Select the training data dir')
    parser.add_argument('--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # Initialize variable arguments to None
    detector = None
    train_path = None
    test_path = None

    # Create the detector
    if vars(args)['detector'] == 'mser':
        detector = MSER_Detector()
    elif vars(args)['detector'] == 'double_equalized_mser':
        detector = Double_Equalized_MSER_Detector()

    # Extract the the train & test paths
    train_path = vars(args)['train_path']
    test_path = vars(args)['test_path']

    if detector is not None and train_path != '' and test_path != '':
        # Load training data
        detector.preprocess_data(train_path, True)

        # Training
        status = detector.fit()

        # DEBUG OR VISUALIZE TRAINING RESULTS
        # visualizer = Detector_Visualizer(training_results)
        # visualizer.setup_directory()
        # visualizer.save_to_dir()

        # Load testing data
        detector.preprocess_data(test_path, False)

        # Testing, see results in ./resultado_imgs/
        detector.predict(status)
    else:
        print("Please introduce the options \"--train_path <train_path> --test_path <test_path> "
              "--detector <detector_name>\" in the command line of the terminal")
