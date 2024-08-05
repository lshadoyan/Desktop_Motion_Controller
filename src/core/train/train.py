import os
import numpy
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import toml

config = toml.load("./configs/config.toml")

class GestureDataset(Dataset):
    """
    Dataset to load mediapipe landmark data from numpy files

    Attributes:
        root_dir (str): Root directory of the dataset
        actions (List[str]): List of action/hand movement classes
        data (List[Tuple[str, int]]): List of (file_path, class_label) tuples
    """
    def __init__(self, root_dir, actions, transform=None):
        """
        Initializes the GestureDataset

        Args:
            root_dir (str): Root directory of the numpy dataset
            actions (List[str]): List of action classes.
            transform (callable, optional): Optional transform to be applied on data
        """
        self.root_dir = root_dir
        self.actions = actions
        self.transform = transform
        self.data = self.load_data()
    
    def load_data(self):
        """
        Loads data paths and labels from the file system

        Returns:
            List[Tuple[str, int]]: A list of tuples, each tuple containing a file path and a label
        """
        data = []
        for class_label, action in enumerate(self.actions):
            class_path = os.path.join(self.root_dir, action)
            for vid_num in range(config["video"]["num_vids"]):
                npy_path = os.path.join(class_path, str(vid_num), config["path"]["np_name"])
                if os.path.exists(npy_path):
                    data.append((npy_path, class_label))
        return data

    def __len__(self):
        """
        Return the number of samples in the dataset

        Returns:
            int: The number of samples in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset

        Args: 
            idx (int): Index of the sample to get

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the data tensor and its label
        """
        npy_path, class_label = self.data[idx]

        data = numpy.load(npy_path)

        data_tensor = torch.tensor(data, dtype=torch.float32)

        return data_tensor, class_label
        
class ActionRecognitionModel(nn.Module):
    """
    LSTM-based model for action recognition

    Attributes:
        lstm (nn.LSTM): The LSTM layer
        fc1 (nn.Linear): Connected layer
        relu (nn.ReLU): ReLU activation function
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        """
        Initialize the ActionRecognitionModel

        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of LSTM hidden state
            output_dim (int): Dimension of output (number of classes)
            num_layers (int, optional): Number of LSTM layers, Defaults to 4
        """
        super(ActionRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.gelu(self.fc1(x))
        return x


def train(model, dataloader, num_epochs, learning_rate = 0.001):
    """
    Trains and saves the model

    Args:
        model (ActionRecognitionModel): The model to train
        dataloader (DataLoader): DataLoader for training data
        num_epochs (int): Number of epochs to train for
        learning_rate (float, optional): learning rate for optimizer. Defaults to 0.001
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')


    print('Finished Training')
    torch.save(model.state_dict(), config["path"]["saved_model"])

def evaluate_model(model, dataloader):
    """
    Evaluates the model's accuracy

    Args:
        model (ActionRecognitionModel): The model to evaluate
        dataloader (DataLoader): DataLoader for test data

    Returns:
        float: Accuracy of the model
    """
    model.eval() 
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print(predicted)
            print(labels)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
