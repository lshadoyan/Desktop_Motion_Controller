import cv2
import mediapipe as mp
import os
from src.core.utils.utils import create_directory, mediapipe_detection, draw_landmarks
import toml

config = toml.load("./configs/config.toml")

def get_center_position(text, font, font_scale, thickness, image):
    """
    Calculates the center for text on an image

    Args:
        text (str): The text to display
        font (int): Font type
        font_scale (float): Scale factor for the font size
        thickness (int): Thickness of the text
        image (np.array): The image to overaly the text

    Returns:
        Tuple[int, int]: (x, y) coordinates for the center position of the text
    """
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    return (text_x, text_y)

def get_top_center_position(text, font, font_scale, thickness, image):
    """
    Calculates the top center for text on an image

    Args:
        text (str): The text to display
        font (int): Font type
        font_scale (float): Scale factor for the font size
        thickness (int): Thickness of the text
        image (np.array): The image to overaly the text
    
    Returns:
        Tuple[int, int]: (x, y) coordinates for the top center position of the text

    """
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10 
    return (text_x, text_y)

def draw_instruction_and_info(image, action, vid_num, frame_num):
    """
    Drawing text on a frame

    Args:
        image (np.ndarray): The image to draw on
        action (str): The current action performed
        vid_num (int): Current video number
        frame_num (int): Current frame number
    """
    top_center_text = f"{action} FRAMES -- VIDEO NUMBER {vid_num}"
    top_center_position = get_top_center_position(top_center_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1, image)
    cv2.putText(image, top_center_text, top_center_position, cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('OpenCV Feed', image)
    
    if frame_num == 0: 
        center_text = 'GET ACTION READY'
        center_position = get_center_position(center_text, cv2.FONT_HERSHEY_DUPLEX, 1, 4, image)
        cv2.putText(image, center_text, center_position, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)
        cv2.waitKey(1000)
            
def extract_frames(actions, num_vids, frames_num, dir_path, test):
    """
    Extract frames from the camera for each action and video

    Args:
        actions (List[str]): List of actions to capture
        num_vids (int): Number of videos per action
        frames_num (int): Number of frames per video
        dir_path (str): Directory path to save the frames
        test (bool): If True, frames not saved (used for testing)
    """
    create_directory(actions=actions, video_num=num_vids, dir_path=dir_path)

    cap = cv2.VideoCapture(config["video"]["webcam_num"])
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for action in actions:
            for vid_num in range(num_vids):
                for frame_num in range(frames_num):

                    success, frame = cap.read()
                    if not success:
                        break
                    
                    image, results = mediapipe_detection(frame, hands)
                    draw_landmarks(image, results)
                    draw_instruction_and_info(action=action, vid_num=vid_num, frame_num=frame_num, image=image)

                    if not test:
                        frame_path = os.path.join(dir_path, action, str(vid_num), f"{frame_num}.jpg")
                        cv2.imwrite(frame_path, frame)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break

    cap.release()
    cv2.destroyAllWindows()