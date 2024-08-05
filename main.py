from src.core.cli.cli import create_parser
from src.core.system.system import GestureRecognitionSystem

def main(): 
    """
    Main function to run the Action Recognition System

    The function parses the command-line arguments and executes the corresponding methods in the GestureRecognitionSystem class based on the specified command
    """
    parser = create_parser()
    args = parser.parse_args()

    system = GestureRecognitionSystem("./configs/config.toml")

    if args.command == "extract_frames":
        if(args.replace):
            system.extract_frames(actions=args.actions, replace=True)
        else:
            system.extract_frames(actions=args.actions, replace=False)
    elif args.command == "process_data":
        system.process_data()
    elif args.command == "train":
        system.train_model()
    elif args.command == "run":
        system.run_prediction()

if __name__ == "__main__":
    main()