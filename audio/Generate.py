import torch 
from torch.utils.data import Dataset,DataLoader
import torchaudio.transforms as T
import torch.nn as nn
import torchaudio
import os 
from ConvertToSpectrogram import SpectrogramConverter
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.functional


class MelSpectrogramDataset(Dataset):
    def __init__(self, audio_dir, converter, desired_size):
        self.audio_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir)]
        self.converter = converter
        self.desired_size = desired_size if isinstance(desired_size, tuple) else (desired_size, desired_size)
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):

        audio_file = self.audio_files[idx]
        mel_spectrogram = self.converter.load_and_convert_to_mel(audio_file)
        mel_spectrogram = F.interpolate(mel_spectrogram.unsqueeze(0), size=self.desired_size, mode='bilinear', align_corners=False)
        mel_spectrogram = mel_spectrogram.squeeze(0)  # Remove the temporary batch dimension
        return mel_spectrogram, audio_file# Add the batch dimension back

desired_size = 128  # or a tuple (128, 128)
converter = SpectrogramConverter()
dataset = MelSpectrogramDataset("../AudioData", converter, desired_size)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)





class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #x = self.flatten(x)
        #x = x.view(x.size(0), -1) 
        #x = self.fc(x)
        return x    
    
class Decoder (nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(16, 64)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=7, stride=1)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
       # x = self.fc(x)
        #x = x.view(-1, 64, 1, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
converter = SpectrogramConverter()
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, _ = data
            #inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


def save_model_params(model):
    torch.save(model.state_dict(),"model.pth")




train_model(model, train_loader, criterion, optimizer, 10)
save_model_params(model)

audio_file = '../Audiodata/PERRO NEGRO.wav'
mel_spectrogram = converter.load_and_convert_to_mel(audio_file)
mel_spectrogram = F.interpolate(mel_spectrogram.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
mel_spectrogram = mel_spectrogram.squeeze(0)

input_data = mel_spectrogram.unsqueeze(0)

with torch.no_grad():
    output = model(input_data)

waveform = torchaudio.functional.istft(output.squeeze(0),n_fft=converter.n_fft,hop_length=converter.hop_length,win_length=converter.win_length,window=converter.window)


# Save the generated audio
torchaudio.save('generated_audio.wav', waveform, sample_rate=converter.sample_rate)