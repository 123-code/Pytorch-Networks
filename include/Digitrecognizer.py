import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import log_softmax, nn, optim

print("---digit recognizer---")


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# loading datasets.
trainset = datasets.MNIST('./Datasets/mnist_train.csv', download=True, train=True, transform=transform)

valset = datasets.MNIST('./Datasets/mnist_test.csv', download=True, train=False, transform=transform)

# loading datasets to dataloader.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# comprueba que el dataloader funciona y se imorta correctamente
for i in trainloader:
    print(i)
    break

# creacion de clase de la red neuronal 
input_size = 784
hidden_sizes = [128,64]
output_size = 10

#nn.linear represents a network layer, applies ReLU at the end of each layer. 
model = nn.Sequential(
    nn.Linear(input_size,hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0],hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1],output_size),
    nn.LogSoftmax(dim=1)
)

print(model)

criterion = nn.NLLLoss()
images,labels = next(iter(trainloader))
images = images.view(images.shape[0],-1)
logps = model(images)
loss = criterion(logps,labels)


