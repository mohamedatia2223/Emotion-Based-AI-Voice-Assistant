import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import noisereduce as nr
import soundfile as sf
from pocketsphinx import LiveSpeech

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
def listen(language="en-US"):
    recognizer = sr.Recognizer()

    # Step 1: Wake Word Detection
    print("🎙️ Say 'Hey' to activate...")
    for phrase in LiveSpeech():  # Continuously listen for the wake word
        if "hey" in str(phrase).lower():  # Wake word detected
            print("🎤 Wake word detected! AI is listening...")
            break

    # Step 2: Capture Audio with Noise Cancellation
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        print("🎤 Say something...")
        audio = recognizer.listen(source, timeout=10)  # Capture audio

        # Convert AudioData to numpy array for noise reduction
        audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
        reduced_noise = nr.reduce_noise(y=audio_data, sr=audio.sample_rate)

        # Save the reduced noise audio to a temporary file
        sf.write("temp.wav", reduced_noise, audio.sample_rate)
        with sr.AudioFile("temp.wav") as source:
            cleaned_audio = recognizer.record(source)  # Load cleaned audio

    # Step 3: Speech Recognition with Multi-Language Support
    try:
        user_input = recognizer.recognize_google(cleaned_audio, language=language)
        print("🗣️ You said:", user_input)

        # Step 4: Detect Emotion
        detected_emotion = detect_emotion()
        emotion_response = f"You look {detected_emotion.lower()}, how can I help?"

        # Step 5: Get AI Response
        full_input = user_input + " " + emotion_response
        response = get_gemini_response(full_input)
        print("🤖 AI:", response)
        speak(response)

    except sr.UnknownValueError:
        print("🤔 Could not understand.")
    except sr.RequestError:
        print("⚠️ Speech recognition error.")

# Language Selection
languages = {
    "English": "en-US",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Chinese": "zh-CN",
    "Arabic (Saudi Arabia)": "ar-SA",
    "Arabic (Egypt)": "ar-EG", 
    
}
print("Select a language:")
for lang in languages:
    print(f"- {lang}")
selected_lang = input("Enter language: ")

if selected_lang in languages:
    listen(language=languages[selected_lang])
else:
    print("⚠️ Unsupported language. Defaulting to English.")
    listen()
