import mediapipe as mp
import os
import cv2
from src.core.utils.utils import mediapipe_detection, create_directory
import numpy as np

def process_frame(hand):
    """
    Processes a single frame to detect hand landmarks

    Args:
        hand (mp.solutions.hand.Hands): Mediapipe Hands object

    Returns:
        np.array: Frame's processed landmarks
    """
    landmarks = np.zeros((21, 3))

    if hand.multi_hand_landmarks:
        hand_landmarks = hand.multi_hand_landmarks[0]
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
    
    return landmarks

def process_video(frames_num, frames_path):
    """
    Processes all frames for a single video given action and video number

    Args:
        frames_num (int): Number of frames in the video
        frames_path (str): Path to video directory containing the frames for one video
    
    Returns:
        np.array: Array containing the landmarks for all frames in the video
    """
    vid_landmarks = []
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        for frame_num in range(frames_num):
            path = os.path.join(frames_path, f"{frame_num}.jpg")
            frame = cv2.imread(path)

            _, hand = mediapipe_detection(image=frame, model=hands)
            data = process_frame(hand=hand)
            data = data.flatten()
            vid_landmarks.append(data)

        return np.array(vid_landmarks)

def process_and_save_all(frames_num, num_vids, actions, frames_dir, numpy_dir, np_name):
    """
    Processes and saves the landmarks for all videos in the dataset

    Args:
        frames_num (int): Number of frames in the video
        num_vids (int): Number of videos per action
        actions (list): List of actions
        frames_dir (string): Path to the video directory
        numpy_dir (string): Path to the numpy dataset containing the video hand landmarks
        np_name (string): Name of the numpy file
    """

    if not os.path.exists(frames_dir):
        print(f"Error: Video directory '{frames_dir}' does not exist.")
        return

    create_directory(actions=actions, dir_path=numpy_dir, video_num=num_vids)

    for action in actions:
        for vid_num in range(num_vids):
            path = os.path.join(frames_dir, action, str(vid_num))
            landmarks = process_video(frames_num=frames_num, frames_path=path)
            npy_path = os.path.join(numpy_dir, action, str(vid_num), np_name)
            np.save(npy_path, landmarks)
            print(f"Saved landmarks for action '{action}', video {vid_num} to {npy_path}")
