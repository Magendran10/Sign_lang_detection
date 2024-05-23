'''PROBLEM STATEMENT:
We aim to Develop a real-time hand gesture recognition system that accurately translates American Sign Language (ASL) gestures into text.
But,here we created a model to identify the english alphabets using cv2,mediapipe.
 '''
# Import the modules like pickle,cv2,numpy,mediapipe
import pickle
import cv2
import numpy as np
import mediapipe as mp

# Load the trained model
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

# Define the labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # Padding to ensure the input size is consistent
        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))

        # If more than 84 features, truncate to 84 features
        data_aux = data_aux[:84]

        # Make a prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Get the bounding box coordinates
        h, w, _ = frame.shape
        x_min, y_min = min(x_), min(y_)
        x_max, y_max = max(x_), max(y_)

        x1, y1 = int(x_min * w), int(y_min * h)
        x2, y2 = int(x_max * w), int(y_max * h)

        # Draw a rectangle around the detected hand and put the predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()