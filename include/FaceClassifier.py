import torch
from torchvision import datasets,transforms
import numpy as np
import pandas as pd
import zipfile
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from PIL import Image

DataTransforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224))
    ]
)


with zipfile.ZipFile('/content/manwomandetection.zip', 'r') as zip_ref:
    zip_ref.extractall('./Datasets')


train_data = datasets.ImageFolder('/content/Datasets/dataset/train', transform=DataTransforms)

test_data = datasets.ImageFolder('/content/Datasets/dataset/test', transform=DataTransforms)

TrainLoader = DataLoader(train_data,batch_size=10,shuffle=True)

TestLoader = DataLoader(test_data,batch_size=10,shuffle=False)

class CNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

epochs = 3

max_trn_batch = 800
max_tst_batch = 300

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
   
    for b, (X_train, y_train) in enumerate(TrainLoader):
        
        
        if b == max_trn_batch:
            break
        b+=1
        
       
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
 
       
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
if b%10 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/8000]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
            


        with torch.no_grad():
          tst_corr = 0  

        for b, (X_test, y_test) in enumerate(TestLoader):
           
            if b == max_tst_batch:
                break

           
            y_val = model(X_test)

           
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()

        
        accuracy = tst_corr.item() / (max_tst_batch * TestLoader.batch_size)

        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)
        print(f"Accuracy: {accuracy:.2%}")
