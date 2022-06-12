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
