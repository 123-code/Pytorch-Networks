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
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])
dataset = datasets.MNIST('./Data', train=True,download=True,transform=dataTransform)
data = datasets

train_loader = DataLoader(dataset,batch_size=8,shuffle=True)

def gen_noise(x,amount):
  noise = torch.rand_like(x)
  amount = amount.view(-1,1,1,1)
  noised_x = (1-amount)*x + amount*noise
  
  return noised_x


amount = torch.linspace(0,1,x.shape[0])
noisedimgs = gen_noise(x,amount)
plt.imshow(make_grid(noisedimgs)[0],cmap='Greys')



class UNet(nn.Module):

  def __init__(self,in_channels=1,out_channels=1):
    super().__init__()

    self.downlayers = torch.nn.ModuleList([
        nn.Conv2d(in_channels,32,kernel_size=5,padding=2),
        nn.Conv2d(32,64,kernel_size=5,padding=2),
        nn.Conv2d(64,64,kernel_size=5,padding=2),
    ])

    self.uplayers = torch.nn.ModuleList([
        nn.Conv2d(64,64,kernel_size=5,padding=2),
        nn.Conv2d(64,32,kernel_size=5,padding=2),
        nn.Conv2d(32,out_channels,kernel_size=5,padding=2),
    ])

    self.activation = nn.SiLU()
    self.downscale = nn.MaxPool2d(2)
    self.upscale = nn.Upsample(scale_factor=2)

  def forward(self,x):
    h = []

    for i,l in enumerate(self.downlayers):
      x = self.activation(l(x))
      if i < 2:
        h.append(x)
        x = self.downscale(x)

    for i,l in enumerate(self.uplayers):
      if i > 0:

        x = self.upscale(x)
        x+=h.pop()
      x = self.activation(l(x))

    return x

net = UNet()
x = torch.rand(8,1,28,28)
net(x).shape


# hyperparameters
batch_size=128
train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
epochs =3

net=UNet()
net.to(device)
criterion = nn.MSELoss()

opt = torch.optim.Adam(net.parameters(),lr=1e-3)
losses = []
# train loop
for epoch in range(epochs):
  for x,y in train_loader:
    x = x.to(device)
    # random noise amounts to images
    noise_amount = torch.rand(x.shape[0]).to(device)
    noised_x = gen_noise(x,noise_amount)
    prediction = net(noised_x)
    loss = criterion(prediction,x)

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(loss.item())


#plotting
n_steps = 5
x = torch.rand(8, 1, 28, 28).to(device) # Start from random
step_history = [x.detach().cpu()]
pred_output_history = []

for i in range(n_steps):
    with torch.no_grad(): # No need to track gradients during inference
        pred = net(x) # Predict the denoised x0
    pred_output_history.append(pred.detach().cpu()) # Store model output for plotting
    mix_factor = 1/(n_steps - i) # How much we move towards the prediction
    x = x*(1-mix_factor) + pred*mix_factor # Move part of the way there
    step_history.append(x.detach().cpu()) # Store step for plotting

fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
axs[0,0].set_title('x (model input)')
axs[0,1].set_title('model prediction')
for i in range(n_steps):
    axs[i, 0].imshow(make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
    axs[i, 1].imshow(make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')

  avg_loss = sum(losses[-len(train_loader):])/len(train_loader)
  print(f"Epoch:{epoch}, loss:{avg_loss}")