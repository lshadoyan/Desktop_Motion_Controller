import argparse

def create_parser():
    """
    Creates an argument parser to create an Action Recognition CLI

    Supports the following commands:
    - extract_frames: Extracts frame/action frames to create a dataset
    - process_data: Processes the frame dataset and saves data as numpy files
    - train: Trains the model
    - run: Runs real-time predictions using the trained models

    Returns:
        argsparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Action Recognition CLI", epilog="For more detailed help on each command, run: main.py <command> --help")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    extract_parser = subparsers.add_parser('extract_frames', help='Extract frames to create a frame dataset')
    extract_parser.add_argument('--actions', nargs='+', default=[], help='List of actions to record')
    extract_parser.add_argument('--replace', action='store_true', help='Replace the existing list of actions')

    subparsers.add_parser("process_data", help="Process the frame dataset and save data as numpy files")

    subparsers.add_parser("train", help="Train the model")

    subparsers.add_parser("run", help="Run real-time predictions using the trained model")

    return parser