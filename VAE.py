import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn 
from flax.training import train_state
import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import optax
from AudioLoader import ConvertToSpectrogram
import os

def ConvertToSpectrogram():

    data_dir = "./Audiodata"
    for file in os.listdir(data_dir):
        audio_file = os.path.join(data_dir, file)
        y, sr = librosa.load(audio_file)
        if y.size > 0:  # Check if audio is not empty
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], 1)
            yield spectrogram
            print("converted to spectrogram")
        else:
            print(f"File {file} is empty, skipping...")

spectrogram_generator = ConvertToSpectrogram()
def get_data_len(data_dir):
    num_files = 0

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        try:
            audio,sr = librosa.load(file_path)
            if audio.size > 0:
                print("loaded file")

                num_files += 1
        except Exception as e:
            print(f"Error loading file:{e}")
            continue

    return num_files

num_files = get_data_len("./Audiodata")


class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self,x):
        encoder = Encoder()
        decoder = Decoder()
        encoded = encoder(x)
        decoded = decoder(encoded)
        return decoded
        
class Encoder(nn.Module):
    @nn.compact
    def __call__(self,x):
        x = nn.Conv(32,kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(64,kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(128,kernel_size=(3,3))(x)
        x = nn.relu(x)
        return x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self,x):
        x = nn.ConvTranspose(128,kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(64,kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(32,kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(1,kernel_size=(3,3))(x)
        x = nn.relu(x)
        return x
        

model = Autoencoder()
init_spectrogram = next(spectrogram_generator)
rng = random.PRNGKey(0)
params = model.init(rng,init_spectrogram)['params']
tx = optax.adam(learning_rate=0.001)
opt_state = tx.init(params)


@jax.jit

def train_step(params,opt_state,batch):
    def loss_function(params):
        predictions = model.apply({'params':params},batch)
        loss = jnp.mean((predictions-batch)**2)
        return loss
    
    grad = jax.grad(loss_function)(params)
    updates,opt_state = tx.update(grad,opt_state,params)
    params = optax.apply_updates(params,updates)
    return params,opt_state

num_epochs = 10

batch_size = 1

for epoch in range(num_epochs):
    
    spectrogram_generator = ConvertToSpectrogram()  # Reset the generator

    for _ in range(num_files):
        spectrogram = next(spectrogram_generator)
        params, opt_state = train_step(params, opt_state, spectrogram)

    print(f'Epoch {epoch+1}')

reconstructed_spectrogram = model.apply({'params':params}, init_spectrogram)
print("Saving model parameters...")
jnp.save("model_params.npy", params)


