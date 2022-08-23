import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

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


class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.flatten = nn.Flatten()
        # creating network layers
        self.l1 = torch.nn.Linear(784, 256)
        self.l2 = torch.nn.Linear(256,10)
        # setting activation functions up
        self.activation = torch.nn.ReLU()
        self.Softmax = torch.nn.Softmax()

# Model's Forward operations.
    def forward(self,x_batch):
        x_batch = self.l1(x_batch)
        x_batch = self.activation(x_batch)
        x_batch = self.l2(x_batch)
        x_batch = self.Softmax(x_batch)
        return x_batch

 
digitrecognizer = NN()

print(digitrecognizer)

        
numbers = torch.randn(1,28,28)
#print(numbers)
Forward_pass = digitrecognizer.forward(numbers)
probability = nn.Softmax(dim=1)
y_pred = probability.argmax(1)
print(f"Predicted:{y_pred}")
print("Funcionó")

'''
train_data_loader = torch.utils.data.DataLoader(mnist_train,
batch_size=1,
shuffle=True)
'''
