import streamlit as st
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import io
import os
from langchain_community.llms import Ollama
import pyttsx3
import speech_recognition as sr
# Set environment variable to handle duplicate libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize WhisperModel and Ollama
model_size = "base.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=5)
llm = Ollama(model="tinyllama")
r=sr.Recognizer()
st.title("J.A.R.V.I.S")    
def record_audio():
    with st.spinner("Recording..."):
        recorded_audio = sd.rec(int(5 * 44100), samplerate=44100, channels=2, dtype="int16")
        sd.wait()
        sf.write("recorded_audio.wav", recorded_audio, samplerate=44100)
    return "recorded_audio.wav"
def transcribe_and_respond(audio_file):
    with open(audio_file, "rb") as audio:
        segments, _ = model.transcribe(io.BytesIO(audio.read()), beam_size=10)
    for segment in segments:
        prompt = segment.text
        st.write(prompt)
    if prompt:
        response = llm.invoke(prompt+'shortly ')
        st.success("Response: " + response)
        pyttsx3.speak(response)
    else:
        st.error("Failed to transcribe audio.")

# Record and transcribe audio
if st.button("Record"):
    audio_file = record_audio()
    transcribe_and_respond(audio_file)

 
