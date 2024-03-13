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

audio_file = './audio.wav'
y,sr = librosa.load(audio_file)
spectrogram = librosa.feature.melspectrogram(y=y,sr=sr)
spectrogram = librosa.power_to_db(spectrogram,ref=np.max)
spectrogram = spectrogram.reshape(1,spectrogram.shape[0],spectrogram.shape[1],1)

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
rng = random.PRNGKey(0)
params = model.init(rng,spectrogram)['params']
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
    params,opt_state = train_step(params,opt_state,spectrogram)
    print(f'Epoch {epoch+1}')

reconstructed_spectrogram = model.apply({'params':params},spectrogram) 



