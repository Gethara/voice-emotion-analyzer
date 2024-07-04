import os
import streamlit as st
from pydub import AudioSegment
from model_training.extract_feature import extract_feature
from model_training.predict import predict
from audiorecorder import audiorecorder
from matplotlib import pyplot as plt
import tempfile

def display_waveform(audio_bytes):
    try:
        audio = AudioSegment.from_mp3data(audio_bytes)
        fig, ax = plt.subplots()
        ax.plot(audio.get_sample_locations(), audio.frame_data)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        fig.suptitle("Recorded Audio Waveform")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred while displaying the waveform: {e}")

def main():
    st.set_page_config(page_title="Audio Emotion Analyzer", layout="wide")
    try:
        with open('styles.css') as f:
            css = f.read()
        st.markdown(f'<style>{css} </style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while loading the styles: {e}")
    
    st.markdown("<h1 style='text-align: center; color: black;'>Audio Emotion Analyzer 🤪</h1>", unsafe_allow_html=True)
    column1, column2, column3 = st.columns([4, 2, 3])

    with column1:
        uploaded_file_placeholder = st.empty()
        success_message = st.empty()
        uploaded_file = uploaded_file_placeholder.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if uploaded_file is not None:
            try:
                audio = AudioSegment.from_file(uploaded_file)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Audio Playback")
                    st.audio(uploaded_file, format='audio/wav')
                with col2:
                    st.subheader("Audio Information")
                    st.write(f"**File name:** {uploaded_file.name}")
                    st.write(f"**Duration:** {len(audio) / 1000:.2f} seconds")
                success_message.success("File uploaded successfully!")
                
                features = extract_feature(uploaded_file, chroma=True, mfcc=True, mel=True)
                emotion = predict(features)
                expressed_emotion = emotion[0][0]
                emotion_list = emotion[2]
                percentage = emotion[1][0]
                st.write(f"Predicted Emotion: {expressed_emotion.title()}")
                
                col3, col4 = st.columns(2)
                with col3:
                    st.write(f"{emotion_list[0].title()} 😡: {round(percentage[0]*100)}%")
                    st.write(f"{emotion_list[1].title()} 😨: {round(percentage[1]*100)}%")
                with col4:
                    st.write(f"{emotion_list[2].title()} 😀: {round(percentage[2]*100)}%")
                    st.write(f"{emotion_list[3].title()} 😢: {round(percentage[3]*100)}%")
            except Exception as e:
                st.error(f"An error occurred while processing the uploaded file: {e}")

    with column2:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.write("You can give voice inputs here:")
            audio = audiorecorder(start_prompt="Start Recording 🔴", stop_prompt="Stop Recording ⏹", show_visualizer=True, key=None)
            
        if len(audio) > 0:
            try:
                temp_file_descriptor, temp_file_path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(temp_file_descriptor, 'wb') as f:
                    f.write(audio.export().read())
                
                with open(temp_file_path, 'rb') as f:
                    audio_bytes = f.read()
                    temp_file_path = f.name

                st.audio(audio_bytes)
                features = extract_feature(temp_file_path, chroma=True, mfcc=True, mel=True)
                emotion = predict(features)
                expressed_emotion = emotion[0][0]
                emotion_list = emotion[2]
                percentage = emotion[1][0]
                st.write(f"Predicted Emotion: {expressed_emotion}")
                st.write(f"{emotion_list[0].title()} 😡: {round(percentage[0]*100)}%")
                st.write(f"{emotion_list[1].title()} 😨: {round(percentage[1]*100)}%")
                st.write(f"{emotion_list[2].title()} 😀: {round(percentage[2]*100)}%")
                st.write(f"{emotion_list[3].title()} 😢: {round(percentage[3]*100)}%")
            except Exception as e:
                st.error(f"An error occurred while processing the recorded audio: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    with column3:
        try:
            st.image("D:\\voice emotion\\emoji2.png")
        except Exception as e:
            st.error(f"An error occurred while loading the image: {e}")

if __name__ == '__main__':
    main()
