<h1 align="center">Welcome to Dual CNN System for Improving Visibility 👋</h1>
<p>
</p>

> Built in collaboration with my teammates Seongjun and Vishnu. Re-uploaded code to my profile for visibility.

A dual CNN system, computer vision-based system designed to assist firefighters' vision during onsite operation through smoke removal and edge detection

## Overview

This project leverages machine learning and real-time image enhancer (with a hardware). By utilizing TensorFlow Lite, the system can run efficiently on low-power devices deployed in the field, instead of relying on cloud connectivity.

## Features

- **Smoke Removal**: Removes or "dehazes" the smoke in an image
- **Edge Detection**: Highlights the edges by converting the RGB image to a Grayscale image with enhanced edge outlining
- **Offline Capability**: Functions without internet connectivity for reliability in remote areas

## Project Structure

```
vision-aid-for-firefighters/
├── edge/                          # Edge device deployment code
├── smoke/                         # Smoke detection models and utilities
├── .gitignore                     # Git ignore configuration
└── tflite_runtime-*.whl          # TensorFlow Lite runtime wheel for ARM devices
```

## Technologies Used

- **Python 3.7+**: Core programming language
- **TensorFlow & Keras**: Primary machine learning Python packages utilized
- **OpenCV**: Image processing and computer vision operations
- **ARM Architecture Support**: Compatible with Raspberry Pi and similar edge devices

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Compatible edge device (Raspberry Pi, Jetson Nano, etc.) or desktop for testing

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/seongjunkang123/vision-aid-for-firefighters.git
   cd vision-aid-for-firefighters
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install TensorFlow Lite runtime** (for ARM devices like Raspberry Pi)
   ```bash
   pip install tflite_runtime-2.11.0-cp37-cp37m-manylinux2014_armv7l.whl
   ```

4. **Verify installation**
   ```bash
   python -c "import tflite_runtime; print('TensorFlow Lite installed successfully')"
   ```

## Usage

### Running Detection on Edge Device

```bash
python edge/test.py 
# choose the function and the image you desire by changing the index from the test images directory
```

### Smoke Detection

```bash
python smoke/test.py 
# choose the function and the image you desire by changing the index from the test images directory
```

## Hardware Recommendations

- **Raspberry Pi 4B** (4GB RAM recommended)
