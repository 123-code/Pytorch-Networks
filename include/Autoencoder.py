import os 
import torch
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

data = datasets.MNIST(root="./data",download=True,transform=transform)


class Encoder(nn.Module):
    def __init__(self,out_channels=1):
        super().__init__()
        
    nn.Sequential(
        nn.Conv2d(1,8,kernel_size=3,padding=1),
        nn.Conv2d(32,8,kernel_size=3,padding=1),
        nn.Conv2d(64,8,kernel_size=3,padding=1),
    )
    
    def forward(self,x):
        
    
    
    
