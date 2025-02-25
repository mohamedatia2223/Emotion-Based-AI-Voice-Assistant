import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set up Gemini API Key
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Load Emotion Detection Model
model_path = r'C:\Users\Asus\Desktop\c\ai\emotion_model.h5'  # Use the correct path
emotion_model = load_model(model_path)

# Emotion Labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to get response from Google Gemini
def get_gemini_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(user_input)
        return response.text if response.text else "I couldn't process your request."
    except Exception as e:
        print(f"⚠️ Gemini API Error: {str(e)}")
        return "There was an error processing your request."

# Function to convert text to speech
def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("start response.mp3")

# Function to capture face and detect emotion
def detect_emotion():
    cap = cv2.VideoCapture(0)
    emotion_counts = {label: 0 for label in emotion_labels}
    
    for _ in range(50):  # Capture frames for approx 5 seconds (assuming 10 FPS)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert frame to grayscale and resize to match model input
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        image = np.expand_dims(resized, axis=-1) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)
        
        # Predict emotion
        prediction = emotion_model.predict(image)
        emotion = emotion_labels[np.argmax(prediction)]
        emotion_counts[emotion] += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Get most frequent emotion detected
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    print(f"🧠 Detected Emotion: {dominant_emotion}")
    return dominant_emotion

# Function to listen to microphone input
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Say 'Hey' to activate...")
        recognizer.adjust_for_ambient_noise(source)
        
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio).lower()
            print("🗣️ You said:", text)

            if "hey" in text:
                print("🎤 AI is listening... Say something...")
                audio = recognizer.listen(source, timeout=10)
                user_input = recognizer.recognize_google(audio)
                print("🗣️ You said:", user_input)
                
                # Detect emotion after greeting
                detected_emotion = detect_emotion()
                emotion_response = f"You look {detected_emotion.lower()}, how can I help?"
                
                # Get AI response
                full_input = user_input + " " + emotion_response
                response = get_gemini_response(full_input)
                print("🤖 AI:", response)
                speak(response)
                
        except sr.UnknownValueError:
            print("🤔 Could not understand.")
        except sr.RequestError:
            print("⚠️ Speech recognition error.")

# Run the voice assistant
listen()
