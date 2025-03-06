from gtts import gTTS
import os

class TTS :
    def __init__(self,lang='ar'):
        self.lang = lang

    def speak(self,text):
        self.tts = gTTS(text=text, lang='ar')

        self.tts.save("dog_voice.mp3")
        os.system("start dog_voice.mp3")
