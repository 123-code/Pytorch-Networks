import torch
import torchaudio
import torchaudio.transforms as transforms
import os

class SpectrogramConverter:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_spec_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def load_and_convert_to_mel(self, audio_file):
        waveform, _ = torchaudio.load(audio_file)
        mel_spectrogram = self.mel_spec_transform(waveform)
        return mel_spectrogram

converter = SpectrogramConverter()

audio_dir = "AudioData"
for file in os.listdir(audio_dir):
    file_path = os.path.join(audio_dir, file)
    mel_spectrogram = converter.load_and_convert_to_mel(file_path)
    print(f"Mel spectrogram shape for {file}: {mel_spectrogram.shape}")