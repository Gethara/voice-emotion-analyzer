# **Audio Emotion Analyzer**

**Introduction**
This project aims to classify emotions from audio recordings using machine learning techniques. It utilizes the RAVDESS and ASVP-ESD datasets for training the model and leverages features like Mel Frequency Cepstral Coefficient **(MFCC), Chroma, and Mel** Spectrogram Frequency.

**Table of Contents**

1.  Project Structure
2.  Datasets
3.  Features Extraction
4.  Model Training
5.  Model Evaluation
6.  Streamlit Application
7.  Installation
8.  Usage
9.  Results
10. Contributing
11. License

    
**Project Structure**     
```python
"""
Audio-Emotion-Analyzer/
│
├── model_training/
│   ├── __init__.py
│   ├── emotions.py
│   ├── extract_feature.py
│   ├── load_data.py
│   ├── model_summary.py
│   ├── predict.py
│
├── audio_files/
│   ├── ...
│
├── styles.css
├── app.py
├── README.md
└── requirements.txt
"""
```
**Datasets**
The project uses the following datasets:

1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
2. ASVP-ESD (Audio Speech Voice Print - Emotional Speech Dataset)
   
We observe only the following emotions:



*   Happy
*   Sad
*   Angry
*   Fearful


 **Features Extraction**
We extract the following features from the audio files:

 

1.**MFCC:** Mel Frequency Cepstral Coefficient, representing the short-term power spectrum of a sound.

2.**Chroma:** Pertains to the 12 different pitch classes.

3.**Mel:** Mel Spectrogram Frequency.

Feature extraction code is available in
`model_training/extract_feature.py`
```pyhton
def extract_feature(file_name, mfcc, chroma, mel, duration=3):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        
        max_length = int(sample_rate * duration)
        if len(X) > max_length:
            X = X[:max_length]
        else:
            X = np.pad(X, (0, max_length - len(X)), mode='constant')
            
        if chroma:
            stft=np.abs(librosa.stft(X, n_fft=16))   # stft = short term fuorier transform
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs.flatten()))
        
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma.flatten()))
        
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel.flatten()))
        return result[:180]
```


 **Model Training**
The model is trained using an **MLP (Multilayer Perceptron) classifier**. 

The training script can be found in `model_training/train_model.py`
```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from model_training.load_data import load_data
from sklearn.preprocessing import StandardScaler
import joblib
from model_training.model_summary import model_summary


x_train,x_test,y_train,y_test = load_data(test_size=0.20)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model=MLPClassifier(alpha=0.01, batch_size=250, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=1500 , random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

joblib_file = f"mlp_classifier_model_with_accuracy_{accuracy*100}.pkl"  
joblib.dump(model, joblib_file)

print(f"Model saved to {joblib_file}")
model_summary(model, y_test,y_pred)

```



 **GridsearchCV**
Here Grid search was used to find the best model.
Following are the details of the best model.


```python
"""
Pipeline(steps=[('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(k=120)),
                ('mlp',
                 MLPClassifier(alpha=0.001, hidden_layer_sizes=(300,),      
                               max_iter=4000, random_state=11, solver='sgd',
                               verbose=True))])
"""
```
Here verbose parameter was set to True inorder to visualize the execution in the terminal.

**Streamlit Application**

A Streamlit app is created to allow users to upload audio files and give voice inputs and then get emotion predictions. The app script is located in `App/main.py`.

```python
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

```

**Installation**
To install the required dependencies, run

`pip install -r requirements.txt`


 **Usage**

1. Model Training:

`python model_training/train_model.py`

2. Streamlit Application:

`streamlit run app/main.py`

**Results**

The model achieves an accuracy of approximately 70% on the test set.
