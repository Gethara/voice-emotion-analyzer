import os, glob
from model_training.emotions import emotions, observed_emotions
from model_training.extract_feature import extract_feature
from sklearn.model_selection import train_test_split
import numpy as np

"""The following function is to Load the data and extract features for each sound file, then
   output Test and Train datasets"""
   
def load_data(test_size=0.2):
    x,y=[],[]
    i=0
    for file in glob.glob("D:\\voice emotion\\audio_speech_actors_01-24\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
        i+=1
        if i>600:
           break    
    
    for file in glob.glob("D:\\voice emotion\\ASVP-ESD-Update\\actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
       
    return train_test_split(np.array(x), y, test_size=test_size, random_state=43)