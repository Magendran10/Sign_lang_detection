# Real-Time Hand Gesture Recognition for American Sign Language (ASL) Translation

## Overview
This project aims to develop a real-time hand gesture recognition system that accurately translates American Sign Language (ASL) gestures into text. American Sign Language is a visual language used by millions of people worldwide, primarily by those who are deaf or hard of hearing. By leveraging computer vision and machine learning techniques, this project facilitates communication between ASL users and non-ASL speakers by translating hand gestures into readable text.

## Features
- Real-time hand tracking: Utilizes computer vision techniques to detect and track hand gestures in real-time.
- Gesture recognition: Employs a machine learning model trained on hand gesture data to recognize ASL gestures.
- Text translation: Translates recognized gestures into corresponding English alphabet characters.
- User-friendly interface: Provides a simple and intuitive interface for users to interact with the system.

## Technologies Used
- OpenCV: Open-source computer vision library for real-time image processing.
- Mediapipe: Google's machine learning framework for building multimodal (e.g., video, audio) applied ML pipelines.
- NumPy: Library for numerical computing with support for arrays and matrices.
- Python: Programming language used for implementing the system and its components.

## Usage
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Run the `hand_gesture_recognition.py` script to start the real-time hand gesture recognition system.
3. Position your hand in front of the camera, and the system will detect and translate your gestures into text.

## Model Training
The machine learning model used for gesture recognition was trained on a dataset of hand gesture images using a convolutional neural network (CNN) architecture. The dataset consisted of labeled images representing various ASL gestures corresponding to English alphabet characters. The model was trained to classify these gestures accurately, achieving high accuracy on the test set.

## Future Enhancements
- Expand gesture recognition: Enhance the system to recognize a wider range of ASL gestures, including words and phrases.
- Improve accuracy: Fine-tune the machine learning model to improve its accuracy and robustness in recognizing gestures.
- Multi-language support: Extend the system to support translation into multiple spoken languages, enabling communication with speakers of different languages.

## Contributors
- Magendran P: Project Lead & Developer
- Gowtham P: Machine Learning Engineer
- Magudesh & Deena: Computer Vision Specialist

## License
This project is licensed under the [MIT License](LICENSE).
