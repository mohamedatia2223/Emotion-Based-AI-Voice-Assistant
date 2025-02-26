import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = r'C:\Users\Asus\Desktop\c\ai\emotion_model.h5'  # Use the correct path
model = load_model(model_path)

# Define the emotion labels (assuming the order of your classes in the model)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral','sad', 'surprise']  # Adjust this list according to your model's classes

# Load and preprocess the image you want to test
test_folder = r'C:\Users\Asus\Desktop\c\ai\dataset\train'

for emotion in emotion_labels:
    emotion_folder = os.path.join(test_folder, emotion)
    
    if not os.path.isdir(emotion_folder):
        continue  # Skip if the emotion folder doesn't exist

    # Iterate over images in the emotion folder
    for filename in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, filename)
        
        if filename.endswith(('jpg', 'jpeg', 'png')):  # Check file extensions
            # Load the image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = cv2.resize(img, (48, 48))  # Resize image to 48x48
            img = img / 255.0  # Normalize the image
            img = np.expand_dims(img, axis=-1)  # Add channel dimension (48x48x1)
            img = np.expand_dims(img, axis=0)  # Add batch dimension (1x48x48x1)

            # Make a prediction
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_emotion = emotion_labels[predicted_class]

            print(f"Predicted emotion for {filename}: {predicted_emotion}")
