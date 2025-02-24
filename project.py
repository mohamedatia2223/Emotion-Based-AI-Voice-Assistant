import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
from TTS.api import TTS
import keyboard
import random  # Import random module
import os

# Load pre-trained emotion detection model
model_path = r'C:\Users\Asus\Desktop\c\ai\fine_tuned_emotion_modelv3.0.h5'
model = tf.keras.models.load_model(model_path)
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

# Emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define multiple responses for each emotion
responses = {
    'Happy': [
        "You’re glowing with happiness! What’s making you smile today?",
        "I love seeing you this happy! Let’s celebrate together!",
        "Your joy is contagious! Keep shining!"
    ],
    'Sad': [
        "I’m sorry you’re feeling down. Remember, it’s okay to feel this way. I’m here for you.",
        "You’re stronger than you think. Let’s take it one step at a time.",
        "It’s okay to cry. I’ll stay by your side until you feel better."
    ],
    'Anger': [
        "I can see you’re upset. Let’s take a moment to breathe and calm down.",
        "Anger is a natural emotion. Let’s figure out how to make things better.",
        "I’m here to listen if you want to talk about what’s bothering you."
    ],
    'Fear': [
        "It’s okay to feel scared. I’m here to help you through this.",
        "You’re not alone. Let’s face this together, one step at a time.",
        "Take a deep breath. You’re stronger than your fears."
    ],
    'Surprise': [
        "Wow, something exciting must have happened! Tell me all about it!",
        "I can see you’re surprised! That’s so interesting—what’s going on?",
        "Surprises can be fun! Let’s enjoy this moment together."
    ],
    'Disgust': [
        "I understand that’s unpleasant. Let’s focus on something more positive.",
        "That doesn’t sound fun. How about we think about something you enjoy instead?",
        "I’m here to help you move past this. Let’s find something better to focus on."
    ],
    'Neutral': [
        "How are you feeling today? I’m here to listen if you need me.",
        "Sometimes it’s okay to feel neutral. Let’s find something to brighten your day!",
        "You seem calm and collected. Is there anything on your mind?"
    ]
}

# Buffer to store emotions
emotion_buffer = []
emotion_capture_time = 5  # Capture duration in seconds

# Function to get the most common emotion
def get_most_common_emotion(emotion_buffer):
    emotion_counts = Counter(emotion_buffer)
    most_common_emotion, count = emotion_counts.most_common(1)[0]
    return most_common_emotion

# Variable to track if capturing is active
capturing = False

while True:
    # Check if "Shift" is pressed to start capturing
    if keyboard.is_pressed("shift"):
        print("Emotion capture started...")
        capturing = True
        start_time = time.time()
        emotion_buffer = []  # Reset emotion buffer

    # Check if "End" is pressed to stop the program
    if keyboard.is_pressed("end"):
        print("Exiting application...")
        break

    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        continue

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_resized = face_resized.astype('float32') / 255
        face_resized = np.expand_dims(face_resized, axis=-1)
        face_resized = np.expand_dims(face_resized, axis=0)

        # Predict emotion
        emotion_probabilities = model.predict(face_resized)
        emotion_index = np.argmax(emotion_probabilities)
        emotion = emotion_labels[emotion_index]

        if capturing:
            emotion_buffer.append(emotion)

            # If 5 seconds have passed, process emotions
            if time.time() - start_time >= emotion_capture_time:
                most_common_emotion = get_most_common_emotion(emotion_buffer)
                print(f"Most common emotion: {most_common_emotion}")

                # Select a random response for the detected emotion
                response = random.choice(responses.get(most_common_emotion, ["I'm here if you need to talk."]))

                # Speak response
                engine.say(f"The emotion detected is {most_common_emotion}. {response}")
                engine.runAndWait()

                capturing = False  # Stop capturing after processing

        # Display detected emotion
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show video frame
    cv2.imshow('Emotion Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
