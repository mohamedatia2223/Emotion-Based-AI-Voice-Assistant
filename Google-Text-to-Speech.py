from gtts import gTTS
import os

# Define playful dog-like text
text = "اهلا يا عسل"

# Convert text to speech with a different accent (optional)
tts = gTTS(text=text, lang='ar')  # Australian English sounds fun!

# Save the output to an MP3 file
tts.save("dog_voice.mp3")

# Play the generated speech (Windows only)
os.system("start dog_voice.mp3")
