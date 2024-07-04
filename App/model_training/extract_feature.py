import numpy as np
import soundfile
import librosa

"""  mfcc, chroma, mel are boolean values
           mfcc: Mel Frequency Cepstral Coefficient and this represents the short-term power spectrum of a sound
           chroma: Pertains to the 12 different pitch classes
           mel: Mel Spectrogram Frequency

"""
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