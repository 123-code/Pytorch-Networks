import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def ConvertToSpectrogram():

    data_dir = "./Audiodata"

    for file in os.listdir(data_dir):
        audio_file = os.path.join(data_dir,file)
        y,sr = librosa.load(audio_file)
        spectrogram = librosa.feature.melspectrogram(y=y,sr=sr)
        spectrogram = librosa.power_to_db(spectrogram,ref=np.max)
        spectrogram = spectrogram.reshape(1,spectrogram.shape[0],spectrogram.shape[1],1)
        return spectrogram