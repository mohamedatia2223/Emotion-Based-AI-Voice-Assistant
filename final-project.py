import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import time
import os
import face_recognition
import pyodbc
import threading
import logging

# Suppress TensorFlow logging
tf.get_logger().setLevel(logging.ERROR)

# 🔑 Replace with your valid API key
genai.configure(api_key="AIzaSyB8B4uBEM1mN-3q9FqF3S0cT5AkY_adXFc")

# Cache responses to avoid redundant API calls
response_cache = {}

# Load emotion detection model
EMOTION_MODEL_PATH = r'C:\Users\Asus\Desktop\c\ai\models\emotion_model_mobilenetv2.h5'
emotion_model = load_model(EMOTION_MODEL_PATH)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Database settings
DB_CONNECTION_STRING = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-7T0M5PQ;DATABASE=Person_Details;Trusted_Connection=yes;"

def load_known_faces():
    """Load known faces from the database using image paths."""
    known_faces = {}
    
    conn = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = conn.cursor()
    
    cursor.execute("SELECT Name, Age, image_path FROM owners")
    for name, age, image_path in cursor.fetchall():
        # Load image from the file path
        if os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            
            # Extract face encodings
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces[name] = (encodings[0], age)  # Store name, encoding, and age
        else:
            print(f"⚠️ الصورة غير موجودة للمستخدم: {name}")
    
    cursor.close()
    conn.close()
    return known_faces

def recognize_face(frame, known_faces):
    """Recognize faces in the frame and return the name and age of the recognized person."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known faces
        matches = face_recognition.compare_faces([data[0] for data in known_faces.values()], face_encoding)
        face_distances = face_recognition.face_distance([data[0] for data in known_faces.values()], face_encoding)
        
        if True in matches:
            best_match_index = matches.index(True)
            name = list(known_faces.keys())[best_match_index]
            age = known_faces[name][1]
            return name, age
    
    return None, None

def detect_emotion(frame):
    """Detect the most common emotion in the frame."""
    emotion_queue = deque(maxlen=50)  # Store emotions for 5 seconds (assuming ~10 FPS)

    # Preprocess the frame for emotion detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]  # Extract face region
        face = cv2.resize(face, (96, 96))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict emotion (suppress TensorFlow logging)
        emotion_prediction = emotion_model.predict(face, verbose=0)
        emotion_index = np.argmax(emotion_prediction)
        emotion = emotion_labels[emotion_index]
        emotion_queue.append(emotion)

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Find the most common emotion
    if emotion_queue:
        most_common_emotion = max(set(emotion_queue), key=emotion_queue.count)
        return most_common_emotion, frame
    else:
        return "Neutral", frame  # Default if no face is detected

def speak(text):
    """Speak using gTTS."""
    output_file = "response.mp3"
    
    # Generate speech
    tts = gTTS(text=text, lang="ar")  # Arabic language
    tts.save(output_file)
    
    # Play the audio file and wait for it to finish
    playsound(output_file)
    
    # Clean up the audio file
    os.remove(output_file)

def get_gemini_response(user_input, emotion=None, name=None, age=None):
    """Fetch a response from Gemini, using cached responses if available."""
    if user_input in response_cache:
        return response_cache[user_input]  # Return cached response
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Construct prompt based on emotion, name, and age
        prompt = f" رد علي باللغة العربية بجمله قصيره"
        if emotion:
            prompt += f" بناءً على شعوري الحالي ({emotion})"
        if name and age:
            prompt += f". أنا {name}، عمري {age} سنة"
        prompt += f": {user_input}"
        
        print(f"Prompt sent to Gemini: {prompt}")  # Debugging print
        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            response_cache[user_input] = response.text  # Cache response
            print(f"Response from Gemini: {response.text}")  # Debugging print
            return response.text

        return "لم أتمكن من معالجة طلبك."
    
    except Exception as e:
        print(f"⚠️ خطأ في Gemini API: {str(e)}")
        return "حدث خطأ أثناء معالجة طلبك."

def listen_and_respond(initial_emotion, name=None, age=None):
    """Listen to user input and respond based on initial emotion and recognized face."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
        print("🎤 يمكنك الآن التحدث مع المساعد. قل 'إيقاف' للخروج.")

        # First response based on initial emotion and recognized face
        print("🤖 المساعد: يتم تحضير الرد الأول بناءً على شعورك...")
        first_response = get_gemini_response("مرحبًا", initial_emotion, name, age)
        print("🤖 المساعد:", first_response)
        speak(first_response)

        while True:
            try:
                print("🎤 تحدث الآن...")
                audio = recognizer.listen(source, timeout=5)  # Reduced timeout for faster response
                user_input = recognizer.recognize_google(audio, language="ar-SA")
                print("🗣️ أنت قلت:", user_input)

                if user_input.lower() in ["إيقاف", "خروج", "توقف"]:
                    print("👋 إنهاء المحادثة.")
                    break

                # Get response without considering emotion
                response = get_gemini_response(user_input, name=name, age=age)
                print("🤖 المساعد:", response)

                # Speak response
                speak(response)

            except sr.UnknownValueError:
                print("🤔 لم أفهمك.")
            except sr.RequestError:
                print("⚠️ خطأ في التعرف على الصوت.")

def camera_feed():
    """Display the camera feed and detect emotions in real-time."""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ خطأ في قراءة الإطار من الكاميرا.")
            break

        # Detect emotion (but don't print it)
        emotion, frame = detect_emotion(frame)

        # Display the frame
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def main():
    """Main function to run the chatbot."""
    try:
        # Load known faces from the database
        print("🤖 المساعد: يتم تحميل الوجوه المعروفة من قاعدة البيانات...")
        known_faces = load_known_faces()

        # Initialize variables for face recognition and emotion detection
        name, age = None, None
        initial_emotion = "Neutral"

        # Capture face and emotion for 5 seconds
        print("🤖 المساعد: يتم التعرف على وجهك وكشف شعورك...")
        start_time = time.time()
        while time.time() - start_time < 5:  # Capture for 5 seconds
            ret, frame = cap.read()
            if not ret:
                print("⚠️ خطأ في قراءة الإطار من الكاميرا.")
                break

            # Recognize face
            if not name or not age:
                name, age = recognize_face(frame, known_faces)
                if name and age:
                    print(f"🤖 المساعد: تم التعرف عليك! أنت {name}، عمرك {age} سنة.")

            # Detect emotion
            emotion, frame = detect_emotion(frame)
            if emotion != initial_emotion:
                print(f"🤖 المساعد: شعورك الحالي هو {emotion}")
                initial_emotion = emotion

            # Display the frame
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Start the chatbot in a separate thread
        print("🤖 المساعد: بدء المحادثة...")
        chatbot_thread = threading.Thread(target=listen_and_respond, args=(initial_emotion, name, age))
        chatbot_thread.start()

        # Keep the camera feed running (without printing emotions)
        camera_feed()

        # Wait for the chatbot thread to finish
        chatbot_thread.join()

    except KeyboardInterrupt:
        print("\n👋 تم إنهاء البرنامج بواسطة المستخدم.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Start the chatbot
if __name__ == "__main__":
    main()
