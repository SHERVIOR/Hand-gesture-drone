# AI-Based Hand Gesture Control for Tello Drone

## Machine Learning Hand Gesture Control Project

### Datasets Used
- **Hand Gesture Dataset** – Collects hand landmarks from different gestures like "open", "fist", "point".

### Algorithms Applied
- **Neural Network** (Keras/TensorFlow)  
  - Built using a fully connected deep neural network for hand gesture classification.

### Technologies
- Python 3.x
- **pandas**, **numpy**
- **TensorFlow / Keras** (for Neural Networks)
- **OpenCV** (for webcam video capture)
- **MediaPipe** (for hand landmark extraction)
- **djitellopy** (for controlling the Tello drone)
- Jupyter Notebook / PyCharm (for development)

## Workflow

### 1. Data Collection
- Use `collect_gesture_data.py` to capture real-time hand gestures with MediaPipe.
- Data includes hand landmarks such as `x`, `y`, and `z` coordinates for each detected landmark, labeled with corresponding gestures (e.g., "open", "fist").

### 2. Data Preprocessing
- **Normalization**: Scaling hand landmarks to a range between 0 and 1.
- **Label Encoding**: Transform gesture labels (e.g., "open", "fist") into integer values for model training.

### 3. Model Training and Evaluation
- **Train-Test Split**: 80-20% for training and testing data.
- **Model**: Train a Neural Network using TensorFlow/Keras to classify hand gestures.
- **Evaluation**: Model is evaluated based on accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: Analyze misclassifications and model performance.

### 4. Real-time Gesture Recognition and Drone Control
- Use `gesture_drone_control_nn.py` to predict gestures from webcam input.
- Control the Tello drone based on gesture predictions (e.g., takeoff, land, move forward).

### 5. Model Deployment
- Load the trained model and label encoder.
- Process hand landmarks in real-time and classify gestures to control drone actions.

## Key Takeaways
- **Data Preprocessing**: Normalizing and encoding the data are crucial for model performance.
- **Gesture Control**: Simple gestures can be used to control drone movements without the need for a traditional controller.
- **Neural Networks**: Offers flexibility for recognizing complex patterns like hand gestures.
- **Real-time Processing**: Real-time performance depends on both the quality of data and model inference speed.

## Requirements
- Python 3.x
- **pandas**, **numpy**
- **TensorFlow** / **Keras**
- **OpenCV**
- **MediaPipe**
- **djitellopy**

## Installation
To get started with the project, you need to install the required dependencies. You can install them using the following command:

```bash
pip install tensorflow opencv-python mediapipe djitellopy scikit-learn
