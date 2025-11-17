# Driver Drowsiness Detection 😴 🚫 🚗

[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/Yash21-04/Final-Submission/blob/master/LICENSE.txt)  [![](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

A real-time computer vision system that automatically detects driver drowsiness using facial landmarks and Eye Aspect Ratio (EAR) analysis. The system triggers an alert when drowsiness is detected, helping prevent accidents caused by fatigue.

## 🎯 Applications

This system is designed for:
- Drivers who operate vehicles for extended periods
- Commercial vehicle operators (trucks, buses, taxis)
- Fleet management systems
- Personal vehicle safety enhancement
- Research in driver safety and fatigue detection

## ✨ Features

- **Real-time Detection**: Processes video stream in real-time from webcam
- **Eye Aspect Ratio (EAR)**: Uses scientifically proven EAR metric for drowsiness detection
- **Visual Feedback**: Draws eye contours for visual confirmation
- **Alert System**: Displays warning messages when drowsiness is detected
- **Lightweight**: Efficient algorithm suitable for resource-constrained environments
- **Customizable Thresholds**: Adjustable sensitivity parameters

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Webcam/Camera
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Yash21-04/Final-Submission.git
cd Final-Submission
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install opencv-python
pip install imutils
pip install dlib
pip install scipy
pip install numpy
```

### Step 4: Download Shape Predictor Model

Download the pre-trained facial landmark detector:

```bash
# Create models directory
mkdir models

# Download shape predictor (68 face landmarks)
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract and place in the models/ directory
```

## 📦 Dependencies

```python
- opencv-python (cv2) - Computer vision and image processing
- imutils - Image processing helper functions
- dlib - Facial landmark detection
- scipy - Scientific computing (distance calculations)
- numpy - Numerical operations
```

## 💻 Usage

### Basic Execution

```bash
python Drowsiness_Detection.py
```

### Controls

- **Press 'q'**: Quit the application
- The system will automatically start detecting faces and monitoring eye aspect ratio

### Configuration Parameters

You can modify these parameters in `Drowsiness_Detection.py`:

```python
thresh = 0.25        # EAR threshold for drowsiness (lower = more sensitive)
frame_check = 20     # Consecutive frames before triggering alert
```

## 🧮 Algorithm Details

### Detection Pipeline

1. **Face Detection**: Uses dlib's frontal face detector to locate faces in the frame
2. **Landmark Prediction**: Identifies 68 facial landmarks using shape predictor
3. **Eye Extraction**: Extracts coordinates for left and right eyes
4. **EAR Calculation**: Computes Eye Aspect Ratio for both eyes
5. **Drowsiness Check**: Monitors EAR over consecutive frames
6. **Alert Trigger**: Displays alert if EAR < threshold for specified frames

### Key Parameters

- **EAR Threshold**: `0.25` - Eyes are considered closed below this value
- **Frame Check**: `20` frames - Number of consecutive frames required to trigger alert
- **Facial Landmarks**: 68-point model from dlib

### Algorithm Flow

```
Capture Frame
    ↓
Convert to Grayscale
    ↓
Detect Face
    ↓
Predict Facial Landmarks
    ↓
Extract Eye Coordinates
    ↓
Calculate EAR
    ↓
EAR < Threshold? → Yes → Increment Counter
    ↓                        ↓
    No                    Counter >= 20? → Yes → ALERT!
    ↓                        ↓
Reset Counter              No → Continue
    ↓
Display Frame
```

## 📁 Project Structure

```
Final-Submission/
│
├── Drowsiness_Detection.py    # Main detection script
├── models/
│   └── shape_predictor_68_face_landmarks.dat  # Facial landmark model
├── readme.md                   # Project documentation
├── venv/                       # Virtual environment (not in repo)
└── requirements.txt            # Python dependencies
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.


```

---

⭐ If you find this project useful, please consider giving it a star!

💡 **Stay Alert, Stay Safe!** 🚗
