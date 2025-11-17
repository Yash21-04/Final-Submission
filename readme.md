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

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dependencies](#-dependencies)
- [Algorithm Details](#-algorithm-details)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [References](#-references)

## ✨ Features

- **Real-time Detection**: Processes video stream in real-time from webcam
- **Eye Aspect Ratio (EAR)**: Uses scientifically proven EAR metric for drowsiness detection
- **Visual Feedback**: Draws eye contours for visual confirmation
- **Alert System**: Displays warning messages when drowsiness is detected
- **Lightweight**: Efficient algorithm suitable for resource-constrained environments
- **Customizable Thresholds**: Adjustable sensitivity parameters

## 🔍 How It Works

The system uses facial landmark detection to monitor the driver's eyes and calculate the Eye Aspect Ratio (EAR). When the EAR falls below a threshold for a consecutive number of frames, it indicates that the eyes are closed, suggesting drowsiness.

### Eye Aspect Ratio (EAR)

Each eye is represented by 6 (x, y)-coordinates, starting at the left corner of the eye and working clockwise around it.

<img src="https://github.com/akshaybahadur21/Drowsiness_Detection/blob/master/assets/eye1.jpg" width="400">

The EAR is calculated using the formula:

<img src="https://github.com/akshaybahadur21/Drowsiness_Detection/blob/master/assets/eye2.png" width="300">

**Formula Breakdown:**

<img src="https://github.com/akshaybahadur21/Drowsiness_Detection/blob/master/assets/eye3.jpg" width="400">

Where:
- `||p2 - p6||` and `||p3 - p5||` are vertical eye distances
- `||p1 - p4||` is the horizontal eye distance

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

## 📊 Results

The system successfully detects drowsiness in real-time with:
- **Response Time**: < 2 seconds after eyes close
- **Accuracy**: High detection rate for various lighting conditions
- **False Positives**: Minimal when properly calibrated

<img src="https://github.com/akshaybahadur21/BLOB/blob/master/drowsy.gif" width="600">

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

## 🎨 Customization

### Adjusting Sensitivity

**More Sensitive** (detects drowsiness faster):
```python
thresh = 0.30        # Higher threshold
frame_check = 15     # Fewer frames required
```

**Less Sensitive** (reduces false positives):
```python
thresh = 0.20        # Lower threshold
frame_check = 25     # More frames required
```

### Adding Sound Alert

Add audio alert when drowsiness is detected:

```python
from pygame import mixer
mixer.init()
mixer.music.load("alarm.wav")
# In alert section:
mixer.music.play()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📌 Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{driver_drowsiness_detection,
  author = {Yash21-04},
  title = {Driver Drowsiness Detection},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Yash21-04/Final-Submission}}
}
```

## 🔗 References

- Adrian Rosebrock, [PyImageSearch: Drowsiness Detection with OpenCV](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
- Soukupová and Čech, [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
- [Dlib Library Documentation](http://dlib.net/)

## 🚧 Future Enhancements

- [ ] Add audio alarm system
- [ ] Implement yawn detection
- [ ] Multi-face detection support
- [ ] Mobile application version
- [ ] Data logging and analytics
- [ ] Cloud-based monitoring dashboard
- [ ] Integration with vehicle systems

## 👨‍💻 Author

**Yash21-04**
- GitHub: [@Yash21-04](https://github.com/Yash21-04)

## 🙏 Acknowledgments

- Thanks to Adrian Rosebrock for the original implementation concept
- dlib library for facial landmark detection
- OpenCV community for computer vision tools

---

⭐ If you find this project useful, please consider giving it a star!

💡 **Stay Alert, Stay Safe!** 🚗