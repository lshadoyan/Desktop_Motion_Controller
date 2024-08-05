from src.core.train.train import train, GestureDataset, ActionRecognitionModel, evaluate_model
from torch.utils.data import DataLoader
from src.core.extraction.extract_data import process_and_save_all
from src.core.extraction.extract_frames import extract_frames
from src.core.load.load import real_time_prediction
from src.core.utils.persistent_array import PersistentClasses

import toml
import torch
import os
import shutil


class GestureRecognitionSystem:
    """
    System for action/gesture recognition, includes frame extraction, data processing, 
    model training, and real-time prediction

    Attributes:
        config (dict): Configuration settings loaded from a TOML file
        actions (list): List of gesture actions
    """
    def __init__(self, config_path):
        """
        Initializes GestureRecognitionSystem

        Args:
            config_path (str): The path to the TOML configuration file
        """
        self.config = toml.load(config_path)
        self.classes = PersistentClasses(self.config["path"]["action_array_dir"])
    
    def _create_model(self, model_path=None):
        """
        Initializes an action recogntion model

        Args:
            model_path (str, optional): Path to a saved model state, if provided the
                                        model's state will be loaded from this file

        Returns:
            ActionRecognitionModel: The initialized/loaded model
            
        """
        model = ActionRecognitionModel(
            input_dim=self.config["params"]["input_dim"],
            hidden_dim=self.config["params"]["hidden_dim"], 
            output_dim=self.classes.size(), 
            num_layers=self.config["params"]["num_layers"])
        
        if model_path:
            model.load_state_dict(torch.load(model_path))

        return model
    
    def extract_frames(self, actions, replace):
        """
        Extract frames to create a dataset for gesture recognition

        Args:
            actions (list): List of actions to record
            replace (bool): If True, the existing list of actions is replaced
        """
        if replace and os.path.exists(self.config["path"]["frames_dir"]):
            shutil.rmtree(self.config["path"]["frames_dir"])
        os.makedirs(self.config["path"]["frames_dir"], exist_ok=True)

        extract_frames(actions = actions,
                    num_vids=self.config["video"]["num_vids"], 
                    frames_num=self.config["video"]["frames_num"], 
                    dir_path=self.config["path"]["frames_dir"], 
                    test=False)
        if replace:
            self.classes.clear()
            
        self.classes.add(actions)
    
    def process_data(self):
        """
        Processes the frame dataset and saves data as numpy files
        """
        process_and_save_all(frames_num=self.config["video"]["frames_num"], 
                            num_vids=self.config["video"]["num_vids"], 
                            actions=self.classes.get_array(), frames_dir=self.config["path"]["frames_dir"], 
                            numpy_dir=self.config["path"]["numpy_dir"], 
                            np_name=self.config["path"]["np_name"])
    
    def train_model(self):
        """
        Trains the action recognition model
        """
        dataset = GestureDataset(root_dir=self.config["path"]["numpy_dir"], 
            actions=self.classes.get_array())


        dataloader = DataLoader(dataset, 
            batch_size=self.config["params"]["batch_size"], 
            shuffle=True)
        model = self._create_model()

        train(model=model, 
              dataloader=dataloader, 
              num_epochs=self.config["params"]["num_epochs"])
        evaluate_model(model=model, dataloader=dataloader)

    
    def run_prediction(self):
        """
        Runs real-time predictions using the trained model
        """
        model = self._create_model(self.config["path"]["saved_model"])
        real_time_prediction(model=model, actions=self.classes.get_array())