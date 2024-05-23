import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import logging

# Suppress TensorFlow Lite warnings (optional)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define data directory
DATA_DIR = './data'

# Initialize data and labels
data = []
labels = []

# Start processing time measurement
start_time = time.time()

# Process each image in the dataset
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img_full_path = os.path.join(DATA_DIR, dir_, img_path)
        img = cv2.imread(img_full_path)
        
        if img is None:
            logging.warning(f"Unable to read image {img_full_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        try:
            results = hands.process(img_rgb)
        except Exception as e:
            logging.error(f"Error processing image {img_full_path}: {e}")
            continue
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

        # Log progress
        logging.info(f"Processed image {img_full_path}")

# Save the data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Cleanup
hands.close()
logging.info("Dataset creation complete.")
logging.info(f"Total processing time: {time.time() - start_time} seconds")
