from src.core.utils.utils import mediapipe_detection, draw_landmarks
from src.core.extraction.extract_data import process_frame
from src.core.load.actions import ActionPredictor, MouseHandler, DesktopSwitcher
import cv2
import mediapipe as mp
import toml

config = toml.load("./configs/config.toml")

def video_dimensions():
    """
    Retrieves the video/image dimensions

    Returns:
        int: The height of the image
        int: The width of the image
    """
    cap = cv2.VideoCapture(config["video"]["webcam_num"])
    frame_height, frame_width = None, None
    if not cap.isOpened():
        print("Error: Could not open camera")
    else:
        ret, frame = cap.read()
        if ret:
            frame_height, frame_width = frame.shape[:2]
        else:
            print("Error: Could not read frame")

    cap.release()
    cv2.destroyAllWindows()

    return frame_height, frame_width

def predictor_initialization(model, actions):
        """
        Initializes the ActionPredictor to predict actions and handle those predicted actions

        Args:
            model (ActionRecognitionModel): The trained action recognition model
            actions (list): List of action labels
        
        Returns:
            ActionPredictor: To predict actions and handle their associated controls
        """
        frame_width, frame_height = video_dimensions()
        mouse_handler = MouseHandler(frame_width=frame_width, frame_height=frame_height)
        desktop_switcher = DesktopSwitcher()
        action_predictor = ActionPredictor(model=model, actions=actions, mouse_handler=mouse_handler, desktop_switcher=desktop_switcher)
        return action_predictor

def real_time_prediction(model, actions):
    """
    Performs real-time action prediction using cv2 webcam input

    Args:
        model (ActionRecognitionModel): The trained model
        actions (list): List of action labels
    """
    action_predictor = predictor_initialization(model, actions)
    cap = cv2.VideoCapture(config["video"]["webcam_num"])

    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            image, hand = mediapipe_detection(frame, hands)
            draw_landmarks(image, hand)

            cv2.imshow('Gesture Recognition', image)

            landmarks = process_frame(hand=hand)
            action_predictor.add_landmarks_and_predict(landmarks)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

