import pyttsx3
import time

engine = pyttsx3.init()
voices = engine.getProperty('voices')
print("Available voices:")
for i, voice in enumerate(voices):
    print(f"{i}: {voice.name} - {voice.id}")

engine.setProperty('voice', voices[1].id)  # try voices[0] or voices[1]

engine.say("Hello bro, this is a simple text to speech test.")
engine.runAndWait()
time.sleep(1)
