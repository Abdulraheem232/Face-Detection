
# Face and Eye Detection with OpenCV

This project demonstrates how to use OpenCV for real-time face and eye detection from a webcam feed. The application utilizes Haar Cascade Classifiers to detect faces and eyes in real-time and draws rectangles around them.

## Requirements

- Python 3.x
- OpenCV 4.x or higher
- NumPy

You can install the required libraries using `pip`:

```bash
pip install opencv-python opencv-python-headless numpy
```

## How It Works

1. **Webcam Feed**: The program captures video from your webcam in real-time.
2. **Face Detection**: Haar Cascade Classifier is used to detect faces in the webcam feed.
3. **Eye Detection**: After detecting a face, the program detects eyes within the face region.
4. **Drawing Rectangles**: The program draws blue rectangles around the detected faces and red rectangles around the detected eyes.
5. **Exit**: Press `q` to exit the program.

## Project Structure

```bash
.
├── face_eye_detection.py  # Main script to run face and eye detection
├── README.md              # This file
└── requirements.txt       # List of dependencies
```

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/Abdulraheem232/Face-Detection.git
    cd Face-Detection
    ```

2. Run the `face_eye_detection.py` script:

    ```bash
    python main.py
    ```

3. The webcam feed will open and display real-time face and eye detection. Press the 'q' key to close the application.

## Code Explanation

### `face_eye_detection.py`

```python
import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load pre-trained Haar Cascade Classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start the webcam feed
while True:
    ret, frame = cap.read()  # Capture each frame
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest (ROI) for the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    # Display the resulting frame with rectangles
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
```

### Key Parts of the Code

- **Face Detection**: `detectMultiScale()` is used to detect faces by analyzing the grayscale version of the webcam feed.
- **Eye Detection**: After detecting a face, another call to `detectMultiScale()` is made on the region containing the face to detect eyes.
- **Webcam Feed**: The webcam feed is captured using `cv2.VideoCapture(0)`, where `0` refers to the default camera.

## Troubleshooting

- If the webcam does not open, ensure that the correct camera is selected and accessible.
- If the face or eye detection does not work accurately, try adjusting the parameters passed to `detectMultiScale()` (such as the scale factor and minNeighbors).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
