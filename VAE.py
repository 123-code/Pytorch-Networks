import jax.numpy as jnp
from jax import random
from flax import linen as nn 
from flax.training import train_state
import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_file = './audio.wav'
y,sr = librosa.load(audio_file)
spectrogram = librosa.feature.melspectrogram(y=y,sr=sr)
spectrogram = librosa.power_to_db(spectrogram,ref=np.max)
spectrogram = spectrogram.reshape(1,spectrogram.shape[0],spectrogram.shape[1],1)

class Autoencoder(nn.Module):
    @nn.compact
    def __init__(self):
        pass
class Encoder(nn.Module):
    def __call__(self,x):
        pass