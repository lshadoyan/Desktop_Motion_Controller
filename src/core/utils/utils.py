import os
import cv2
import mediapipe as mp
import math

def create_directory(video_num, actions, dir_path):
    """
    Creates directories for storing video frames and numpy files

    Args:
        video_num (int): Number of videos for each action
        actions (list): List of action labels
        dir_path (string): Path to the directory
    """
    main_path = os.path.join(dir_path)
    for action in actions:
        for num in range(video_num):
            try:
                os.makedirs(os.path.join(main_path, action, str(num)))
            except:
                pass

def mediapipe_detection(image, model):
    """
    Performs hand detection on the given image

    Args:
        image (np.ndarray): The input image
        model (mp.solutions.hands.Hands): Mediapipe hands model
    
    Returns:
        np.ndarray: The processed image
        mediapipe.python.solutions.hands.Hands: The results of the detection
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    """
    Draws hand landmarks on the input frame

    Args:
        image (np.ndarray): Input image
        results (mediapipe.python.solutions.hands.Hands): The detection results
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

def midpoint(p1, p2):
    """
    Calculates the midpoint between two points

    Args:
        p1 (list): The first point x, y
        p2 (list): The second point x, y
    
    Returns:
        list: The midpoint x ,y
    """
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def distance(point1, point2):
    """
    Calculates the euclidean distance between two points

    Args:
        point1 (float): The first point
        point2 (float): The second point
    
    Returns:
        float: The distance between the points
    """
    return math.sqrt((point1 - point2) ** 2)