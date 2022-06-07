import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

print("---digit recognizer---")

train_data = datasets.mnisttrain = datasets.MNIST(root='./Datasets/mnist_train.csv'
, train=True,
 download=True,
 transform=ToTensor())

test_data = datasets.mnisttest = datasets.MNIST(root='./Datasets/mnist_test.csv'
, train=False,
 download=True,
 transform=ToTensor())

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloasder = DataLoader(test_data, batch_size=64, shuffle=True)

# comprueba que el dataloader funciona y se imorta correctamente
for i in train_dataloader:
    print(i)
    break

# creacion de clase de la red neuronal 
class NN(nn.Module):
    def __init__(self):
        super.__init()



print("Funcionó")

'''
train_data_loader = torch.utils.data.DataLoader(mnist_train,
batch_size=1,
shuffle=True)
'''