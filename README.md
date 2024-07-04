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

def extract_feature(file_name, mfcc, chroma, mel, duration=3):
    # Implementation here



 **Model Training**
The model is trained using an **MLP (Multilayer Perceptron) classifier**. 

The training script can be found in `model_training/train_model.py`
```python
from sklearn.neural_network import MLPClassifier
# Training code here
```



 **Model Evaluation**
The model is evaluated using accuracy metrics. 
The evaluation script can be found in `model_training/evaluate_model.py`


```python
from sklearn.metrics import accuracy_score
# Evaluation code here
```

**Streamlit Application**

A Streamlit app is created to allow users to upload audio files and get emotion predictions. The app script is located in `app.py`.

```python
import streamlit as st
# Streamlit app code here
```

**Installation**
To install the required dependencies, run

pip install -r requirements.txt


 **Usage**

1. Model Training:

python model_training/train_model.py

2. Streamlit Application:

streamlit run app.py

**Results**

The model achieves an accuracy of approximately 70% on the test set.
