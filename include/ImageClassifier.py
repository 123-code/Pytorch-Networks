import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# data transformations
transformations = transforms.Compose([
 transforms.resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]
])

# import data using torchvision.datasets
train_batch = datasets.ImageFolder("root/label/train",transform=transformations)
val_batch = datasets.ImageFolder("root/label/valid",transform=transformations)
   
# DataLoader to batch data
train_loader = torch.utils.data.DataLoader(train_batch,batch_size=32,shuffle=True)
va_loader = torch.utils.data.DataLoader(val_batch,batch_size=32,shuffle=True)

# model creation 
model = models.densenet161(pretrained=True)
for param in model.parameters():
    param.requires_grad=False