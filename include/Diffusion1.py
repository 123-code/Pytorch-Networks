import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import os
import shutil

dataTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])
dataset = datasets.FGVCAircraft('./Data12', split='train',download=True,transform=dataTransform)
data = datasets

train_loader = DataLoader(dataset,batch_size=10,shuffle=True)


def gen_noise(x,amount):
    noise = torch.rand_like(x)
    noisy_image = (1-amount)*x + amount*noise
    amount = amount.view(-1, 1, 1, 1)
    return noisy_image


x, y = next(iter(train_loader))

amount = torch.linspace(0, 1, x.shape[0])
amount = amount.view(-1, 1, 1,1)
noised_images = gen_noise(x, amount)
plt.imshow(make_grid(noised_images)[0], cmap='Greys');
