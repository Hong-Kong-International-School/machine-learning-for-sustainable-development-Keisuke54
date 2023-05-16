# brew install ffmpeg
# pip install Whisper  

import whisper

model = whisper.load_model("base")

result = model.transcribe("phiTest.mp3")

print(result["text"])