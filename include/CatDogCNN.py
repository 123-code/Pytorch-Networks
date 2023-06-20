import torch 
from torchvision import datasets,transforms 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image  
import matplotlib.pyplot  as plt
import zipfile


with zipfile.ZipFile('/content/dogs-vs-cats-redux-kernels-edition.zip', 'r') as zip_ref:
    zip_ref.extractall('./Datasets')

with zipfile.ZipFile('/content/Datasets/train.zip', 'r') as zip_ref:
    zip_ref.extractall('./Train') 

with zipfile.ZipFile('/content/Datasets/test.zip', 'r') as zip_ref:
    zip_ref.extractall('./Test') 
    
    
   
    
 dataTransform = transforms.Compose(
   [ transforms.ToTensor(),
    transforms.Resize((224,224)),
     
     ]
)
Train_dataset = datasets.ImageFolder('/content/Train',transform=dataTransform)
Test_dataset = datasets.ImageFolder('/content/Test',transform=dataTransform)


Train_Loader = DataLoader(Train_dataset,batch_size=10,shuffle=True)
Test_Loader = DataLoader(Test_dataset,batch_size=10,shuffle=True)
#plotting 

batch = next(iter(Train_Loader))
images, labels = batch[:10]
grid = make_grid(images)
grid_np = grid.numpy().transpose((1, 2, 0))
plt.imshow(grid_np)
plt.axis('off')
plt.show()

# Model

class ConvolutionalNetwork(nn.Module):
  def __init__(self,X):
    super().__init__()

    self.conv1 = nn.Conv2d(3,8,3,1)
    self.conv2 = nn.Conv2d(8,16,3,1)
    self.fc1 = nn.Linear(16*28*28,120)
    self.fc2 = nn.Linear(120,80)
    self.fc3 = nn.Linear(80,2)

  def forward(self,X):
    X = F.ReLU(self.conv1())
    X = F.max_pool2d(X,2,2)
    X = F.ReLU(self.conv2())
    X = F.max_pool2d(X,2,2)
    
