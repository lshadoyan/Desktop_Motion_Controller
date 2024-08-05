# Desktop Motion Controller

This is a machine-learning based tool that allows you to control your Windows desktop using hand gestures. Leveraging OpenCV, MediaPipe, and a PyTorch LSTM model, this project enables cursor control and desktop navigation.

## Installation

### Clone the Repository

```sh
git clone <repo_url>
```

### Primary Method

1. **Build and Run**

   To set up and run real-time processing using the `Makefile`, use:

   ```sh
   make run
   ```

### Alternative Method

1. **Install Dependencies**:

   If you prefer not to use the `Makefile`, install the necessary dependencies directly:

   ```sh
   pip install -r requirements.txt
   ```

2. **Run Real-Time Processing**

   Execute the program with:

   ```sh
   python main.py run
   ```

## Usage

### Gestures

- Pinch fingers, move them left to right: Switch to the left desktop
- Pinch fingers, move them right to left: Switch to right desktop
- Pinch thumb and pointer finger, move them up/down: Scroll up and down
- Pinch thumb and pointer finger: Mouse left click
- Pinch thumb and pointer finger and hold: Mouse right click

## Key Features

### Frame Extraction

- Extract video frames for specified actions.

  ```sh
  make extract ACTIONS="example_action1 example_action2"
  ```

  Process data and train the model:

  ```sh
  make process train
  ```

## Future Functionality

- Enhanced click recognition, potentially integrated into the classifier
