from src.core.utils.utils import midpoint, distance
import torch.nn.functional as F
import numpy as np
import torch
import toml
import math
import pyautogui

config = toml.load("./configs/config.toml")

class ActionPredictor():
    """
    A class to predict actions based on input landmarks ActionRecognitionModel

    Attributes:
        THRESHOLD (float): Threshold value to determine if the hand is open
        model (ActionRecognitionModel): The trained action recognition model
        actions (list): List of action labels
        sequence (list): List to store sequences of landmarks
        mouse_handler (MouseHandler): Instance of MouseHandler to control the mouse
        desktop_switcher (DesktopSwitcher): Instance of DesktopSwitcher to switch desktops
    """

    THRESHOLD = config["ACTIONS"]["threshold"]

    def __init__(self, model, actions, mouse_handler, desktop_switcher):
        """
        Initializes the ActionPredictor with the given model, actions, mouse handler and desktop handler

        Args:
            model (ActionRecognitionModel): The trained action recognition model
            actions (list): List of action labels
            mouse_handler (MouseHandler): Instance of MouseHandler to control the mouse
            desktop_switcher (DesktopSwitcher): Instance of DesktopSwitcher to switch desktops
        """
        self.model = model
        self.actions = actions
        self.sequence = []
        self.mouse_handler = mouse_handler
        self.desktop_switcher = desktop_switcher
    
    def add_landmarks_and_predict(self, landmarks):
        """
        Predicts the action based on the given sequence and appends the current landmarks to the sequence

        Args:
            landmarks (np.ndarray): Array of landmarks for the current frame
        """

        self.sequence = self.sequence[-30:]
        if(len(self.sequence) == 30):
            predicted_label, confidence = self._get_prediction()
            if confidence.item() > config["params"]["confidence"]:
                print(f"Predicted: {predicted_label} (Confidence: {confidence.item():.4f})")
                self._handle_predicted_action(predicted_label=predicted_label, landmarks=landmarks)
        self.sequence.append(landmarks.flatten())

    def _handle_predicted_action(self, predicted_label, landmarks):
        """
        Handles the action based on the predicted label

        Args:
            predicted_label (string): The predicted action label
            landmarks (np.ndarray): Array of landmarks for the current frame
        """
        if predicted_label == config["ACTIONS"]["mouse_action_name"]:
            self.mouse_handler.move_mouse(landmarks, self._is_hand_open(landmarks[8], landmarks[4]))
        elif predicted_label == config["ACTIONS"]["window_action_name"]:
            self.desktop_switcher.switch_desktop(landmarks[8][0])
            self.sequence = []
    
    def _get_prediction(self):
        """
        Predicts the action from the sequence of landmarks

        Args:
            sequence (list): List of landmarks for each frame
            model (ActionRecognitionModel): The trained model
            actions (list): List of action labels
        
        Returns:
            str: Predicted action label
            float: Confidence score
        """
        numpy_array = np.array(self.sequence)
        tensor_data = torch.tensor(numpy_array, dtype=torch.float32)
        tensor_data = tensor_data.view(1, 30, 63)
        with torch.no_grad():
            outputs = self.model(tensor_data)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            predicted_label = self.actions[predicted_class.item()]
        
        return predicted_label, confidence

    def _is_hand_open(self, pointer_finger_tip, thumb_tip):
        """
        Determines if the hand is open based on the distance between the pointer finger tip and the thumb tip

        Args:
            pointer_finger_tip (np.ndarray): x, y, z coordinates of the pointer finger tip
            thumb_top (np.ndarray): x, y, z coordinates of the thumb tip

        Returns:
            bool: True if the hand is open, False otherwise
        """
        distance = math.sqrt(
            (pointer_finger_tip[0] - thumb_tip[0]) ** 2 +
            (pointer_finger_tip[1] - thumb_tip[1]) ** 2 +
            (pointer_finger_tip[2] - thumb_tip[2]) ** 2
        )
        return distance > self.THRESHOLD
    
class MouseHandler():
    """
    A class to handle mouse movements based on input landmarks

    Attributes:
        DEADZONE (float): Threshold value for smoothing control
        SCALE_FACTOR (float): Factor to scale the mouse movement
        MARGIN (float): Margin for mapping to the display
        SCROLL_MULTIPLIER (int): Multiplier for scrolling amount
        prev_x (int): Previous x-coordinate of the mouse
        prev_y (int): Previous y-coordinate of the mouse
        prev_open (bool): Flag to indicate if the hand was open in the previous frame
        prev_y_scroll (int): Previous scroll amount
        right_counter (int): Counter for right-clicks
        frame_width (int): Width of the frame
        frame_height (int): Height of the frame
    """
    DEADZONE = config["ACTIONS"]["deadzone"]
    SCALE_FACTOR = config["ACTIONS"]["scale_factor"]
    MARGIN = config["ACTIONS"]["margin"]
    SCROLL_MULTIPLIER = config["ACTIONS"]["scroll_multiplier"]

    def __init__(self, frame_width, frame_height):
        """
        Initializes the MouseHandler object with the given frame width and height

        Args:
            frame_width (int): Width of the camera frame
            frame_height (int): Height of the camera frame
        """
        self.prev_x = None
        self.prev_y = None
        self.prev_open = True
        self.prev_y_scroll = None
        self.right_counter = 0
        self.frame_width = frame_width
        self.frame_height = frame_height


    def move_mouse(self, landmarks, curr_open):
        """
        Moves the mouse based on the given landmarks and handles mouse clicks and scrolling

        Args:
            landmarks (np.ndarray): Mediapipe landmarks of the hand in the current frame
            curr_open (bool): Whether the hand is currently open or not
        """

        x, y = midpoint(landmarks[8], landmarks[4])
        y_dist = 0
        if x != 0 and y != 0:
            x, y, y_dist = self._smoothing_control(x, y, y_dist)
            x, y = self._map_to_display(x, y)

            scroll_amount = self._calculate_scoll_amount(y)
            self._handle_clicks(curr_open, scroll_amount, y_dist)

            self.prev_open = curr_open
            pyautogui.moveTo(x, y)

    def _smoothing_control(self, x, y, y_dist):
        """
        Smoothens the mouse movement to avoid too much jitter

        Args:
            x (float): The x-coordinate of the mouse (0.0 - 1.0)
            y (float): Y coordinate of the mouse (0.0 - 1.0)
            y_dist (float): Distance moved in the y-direction

        Returns:
            float: The x-coordinate of the mouse (0.0 - frame_width)
            float: The y-coordinate of the mouse (0.0 - frame_height)
            y_dist: The distance the y-coordinate has moved last 
        """
        x, y = ((1-x) * self.frame_width), (y * self.frame_height)

        if self.prev_x is not None and self.prev_y is not None:
            y_dist, x_dist = distance(y, self.prev_y), distance(x, self.prev_x)
            
            if x_dist > self.DEADZONE:
                x = ((x) * self.SCALE_FACTOR) + (self.prev_x * (1 - self.SCALE_FACTOR))
            else:
                x = self.prev_x

            if y_dist > self.DEADZONE:
                y = (y * self.SCALE_FACTOR) + (self.prev_y * (1 - self.SCALE_FACTOR))
            else:
                y = self.prev_y
        
        self.prev_x, self.prev_y = x, y

        return x, y, y_dist

    def _map_to_display(self, x, y):
        """
        Maps the camera/image coordinates to the display dimensions

        Args:
            x (float): The x-coordinate of the mouse (0.0 - frame_width)
            y (float): The y-coordinate of the mouse (0.0 - frame_height)

        Returns:
            int: The x-coordiante of the mouse (0 - screen_width)
            int: The y-coordinate of the mouse (0 - screen_height)
        """
        screen_width, screen_height = pyautogui.size()
        box_x = int(self.frame_width * self.MARGIN)
        box_y = int(self.frame_height * self.MARGIN)
        box_width = int(self.frame_width * (1 - 2 * self.MARGIN))
        box_height = int(self.frame_height * (1 - 2 * self.MARGIN))

        x = int(((x - box_x) / box_width) * screen_width)
        y = int(((y - box_y) / box_height) * screen_height)

        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))

        return x, y
    
    def _calculate_scoll_amount(self, y):
        """
        Calculates the amount to scroll based on the current and previous positions

        Args:
            y (int): The y-coordinate of the current mouse position
        
        Returns:
            int: The amount to scroll, either positive or negative
        """
        scroll_amount = 0
        if self.prev_y_scroll is not None:
            scroll_amount = (y - self.prev_y_scroll) * self.SCROLL_MULTIPLIER 
        
        self.prev_y_scroll = y

        return scroll_amount
    
    def _handle_clicks(self, curr_open, scroll_amount, y_dist):
        if not curr_open:
            if scroll_amount != 0 and (y_dist > 8):
                pyautogui.scroll(int(scroll_amount))
                self.right_counter = 0
            elif self.prev_open and self.right_counter < 10:
                pyautogui.click(button="left")
                self.right_counter = 0
            elif self.right_counter >= 10:
                pyautogui.click(button="right")
                self.right_counter = 0
            self.right_counter += 1


class DesktopSwitcher:
    """
    Class to handle desktop switching based on pointer position
    """
    def __init__(self):
        """
        Initializes the DesktopSwitcher
        """
        pass
    
    def _side(self, pointer_position):
        """
        Determines the side based on the pointer position

        Args:
            pointer_position (float): Position of the pointer (0.0 to 1.0)
        """
        return 'right' if pointer_position >= 0.5 else 'left'

    def switch_desktop(self, pointer_position):
        """
        Switches desktop based on the direction

        Args:
            pointer_position (float): Position of the pointer (0.0 to 1.0)
        """
        direction = self._side(pointer_position)
        pyautogui.hotkey('ctrl', 'win', direction)

