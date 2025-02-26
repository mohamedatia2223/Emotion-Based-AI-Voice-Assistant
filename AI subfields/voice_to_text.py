import speech_recognition as sr
from gtts import gTTS
import os

# Specify your microphone name
MIC_NAME = "Microphone (Realtek(R) Audio)"  # Change if needed

def speak_arabic(text):
    """Converts Arabic text to speech using gTTS and plays it."""
    tts = gTTS(text=text, lang="ar")
    tts.save("response_ar.mp3")
    os.system("start response_ar.mp3")  # Windows: 'start', Mac: 'open', Linux: 'xdg-open'

def main():
    recognizer = sr.Recognizer()
    mic_index = None

    # Find the correct microphone index
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        if MIC_NAME in name:
            mic_index = i
            break

    if mic_index is None:
        print(f"تعذر العثور على الميكروفون: {MIC_NAME}")
        return

    print(f"يتم استخدام الميكروفون: {MIC_NAME} (فهرس: {mic_index})")

    with sr.Microphone(device_index=mic_index) as source:
        print("سأستمع باستمرار... قل 'توقف عن الاستماع' لإنهاء البرنامج.")
        recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                print("الاستماع...")
                audio = recognizer.listen(source, phrase_time_limit=5)  # Listen up to 5 seconds
                text = recognizer.recognize_google(audio, language="ar-EG")

                print("قلت:", text)

                # Stop if user says "توقف عن الاستماع"
                if "توقف عن الاستماع" in text.lower():
                    print("إيقاف البرنامج...")
                    speak_arabic("إلى اللقاء!")
                    break

                # Otherwise, respond by repeating what was said
                response = f"لقد قلت: {text}"
                speak_arabic(response)

            except sr.UnknownValueError:
                print("عذراً، لم أفهم ما قلت.")
            except sr.RequestError:
                print("تعذر الاتصال بالخدمة، تحقق من الاتصال بالإنترنت.")

if __name__ == "__main__":
    main()
