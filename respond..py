import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import os

# Set up Gemini API Key
GEMINI_API_KEY = "AIzaSyB8B4uBEM1mN-3q9FqF3S0cT5AkY_adXFc"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

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
    os.system("start response.mp3")  # Works on Windows

# Function to listen to microphone input
def listen():
    recognizer = sr.Recognizer()

    # Use the default microphone
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

                response = get_gemini_response(user_input)
                print("🤖 AI:", response)
                speak(response)

        except sr.UnknownValueError:
            print("🤔 Could not understand.")
        except sr.RequestError:
            print("⚠️ Speech recognition error.")

# Run the voice assistant
listen()