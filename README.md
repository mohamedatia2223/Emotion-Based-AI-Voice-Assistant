# Emotion-Based AI Voice Assistant

## 📌 Project Overview
This project is an AI-powered voice assistant that interacts with users through speech and detects their emotions using a trained deep-learning model. The assistant listens for a wake word ("Hey"), processes user speech, detects emotions from their facial expressions, and responds accordingly. The response is tailored based on the detected emotion, providing a personalized interaction.

## 🚀 Features
- **Speech Recognition**: Listens and transcribes user speech using `speech_recognition`.
- **Emotion Detection**: Captures a user's facial expression for 5 seconds, processes the most common emotion, and adapts the AI's response.
- **AI-Powered Responses**: Uses Google Gemini API to generate context-aware responses.
- **Text-to-Speech (TTS)**: Converts responses into speech using `gTTS`.
- **Deep Learning Model**: Uses a Convolutional Neural Network (CNN) trained on the **FER-2013 dataset** for emotion detection.

## 🛠️ Installation
### Prerequisites
Ensure you have the following installed:

- Python 3.x
- Required Python libraries:
  ```sh
  pip install google-generativeai speechrecognition gtts tensorflow opencv-python numpy
  ```
- A trained emotion detection model (`emotion_detection.h5`)

### Clone the Repository
```sh
git clone https://github.com/your-username/emotion-voice-assistant.git
cd emotion-voice-assistant
```

## 📄 How It Works
1. The assistant listens for the wake word ("Hey").
2. After activation, it records the user's speech and processes the text.
3. Simultaneously, the system captures video frames for 5 seconds and detects the most common facial expression.
4. The detected emotion is passed into the chatbot.
5. The chatbot formulates an emotion-based response (e.g., "You look sad, how can I cheer you up?").
6. The response is spoken aloud using TTS.

## 📌 Usage
Run the assistant by executing:
```sh
python main.py
```
Ensure your webcam and microphone are enabled for proper functionality.

## 🤖 Future Improvements
- Enhance emotion classification accuracy
- Add real-time sentiment analysis
- Integrate a more advanced NLP model for responses

## 📜 License
This project is licensed under the MIT License.

