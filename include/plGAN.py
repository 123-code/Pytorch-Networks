import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split
from torch.nn import LayerNorm
from torchvision.datasets import MNIST
from torch.nn import GroupNorm

import matplotlib.pyplot as plt
import pytorch_lightning as pl

random_seed = 42
torch.manual_seed(random_seed)

batch_size=128
avail_gpus = min(1,torch.cuda.device_count())
num_workers = int(os.cpu_count()/2)

class MNISTDataModule(pl.LightningDataModule):
  def __init__(self,data_dir='./data',batch_size=batch_size,num_workers=num_workers):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers

    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

  def prepare_data(self):
    MNIST(self.data_dir,train=True,download=True)
    MNIST(self.data_dir,train=False,download=True)

  def setup(self,stage=None):
    if stage == 'fit' or stage is None:
      mnist_full = MNIST(self.data_dir,train=True,transform=self.transform)
      self.mnist_train,self.mnist_val = random_split(mnist_full,[55000,5000])

    if stage == 'test' or stage is None:
      self.mnist_test = MNIST(self.data_dir,train=False,transform=self.transform)

  def train_dataloader(self):
    return DataLoader(self.mnist_train,batch_size=self.batch_size,num_workers=self.num_workers)


  def val_dataloader(self):
    return DataLoader(self.mnist_val,batch_size=self.batch_size,num_workers=self.num_workers)


  def test_dataloader(self):

    return DataLoader(self.mnist_test,batch_size=self.batch_size,num_workers=self.num_workers)


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,10,kernel_size=5)
    self.group_norm1 = GroupNorm(10, 10)
    self.conv2 = nn.Conv2d(10,20,kernel_size=5)
    self.group_norm2 = GroupNorm(20, 20)
    self.fc1 = nn.Linear(320,50)
    self.fc2 = nn.Linear(50,1)


  def forward(self,x):
    x = F.relu(self.group_norm1(self.conv1(x)))
    x = F.relu(self.group_norm2(self.conv2(x)))
    # flatten tensor and feed to the fc layers
    x= x.view(-1,64,7,7)
    x = F.relu(self.fc1(x))
    x = F.dropout(x,training=self.training)
    x = self.fc2(x)
    return x


class Generator(nn.Module):
  def __init__(self,latent_dim):
    super().__init__()
    self.lin1 = nn.Linear(latent_dim,7*7*64)
    self.transpose1 = nn.ConvTranspose2d(64,32,4,stride=2)
    self.transpose2 = nn.ConvTranspose2d(32,16,4,stride=2)
    self.conv = nn.Conv2d(16,1,kernel_size=7)

  def forward(self,x):
    x=self.lin1(x)
    x = F.relu(x)
    #x= x.view(-1,64,7,7)

    x = self.transpose1(x)
    x = F.relu(x)

    x = self.transpose2(x)
    x = F.relu(x)

    return self.conv(x)



class GAN(pl.LightningModule):
  def __init__(self,latent_dim=100,lr=0.0002):
    super().__init__()
    self.save_hyperparameters()
    self.automatic_optimization = False

    self.generator = Generator(latent_dim=self.hparams.latent_dim)
    self.discriminator = Discriminator()
    # random noise to use as generator input

    self.random_noise = torch.randn(6,self.hparams.latent_dim)

  def forward(self,x):
    return self.generator(x)

  def adversarial_loss(self, predicted, target):

    loss = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for GANs
    return loss(predicted, target)
    #return loss.item()

  def training_step(self, batch):
    opt_g, opt_d = self.optimizers()

    #real images, labels
    real_imgs, _ = batch
    z = torch.randn(real_imgs.shape[0],self.hparams.latent_dim)
    z = z.type_as(real_imgs)

    for i in range(1):


      self.toggle_optimizer(opt_g)
      valid = torch.ones(real_imgs.size(0), 1)
      valid = valid.type_as(real_imgs)
      g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
      opt_g.zero_grad()
      self.manual_backward(g_loss)
      opt_g.step()
      self.untoggle_optimizer(opt_g)

      log_dict = {"g_loss":g_loss}
      return {"loss":g_loss,"progress_bar":log_dict,"log":log_dict}


    # train the critic


    for i in range(5):

      valid = torch.ones(self.imgs.size(0), 1)
      valid = valid.type_as(self.imgs)
      self.toggle_optimizer(opt_d)
      # calculates how well the model predicts real images as valid
      real_loss = self.adversarial_loss(self.discriminator(real_imgs),valid)
      fake = torch.zeros(self.imgs.size(0), 1)
      fake = fake.type_as(self.imgs)
      fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)
      opt_d.zero_grad()
      self.manual_backward(self.d_loss)
      opt_d.step()
      self.untoggle_optimizer(opt_d)
      log_dict = {"d_loss":self.d_loss}
      return {"loss":self.d_loss,"progress_bar":log_dict,"log":log_dict}

  def configure_optimizers(self):
    lr = self.hparams.lr
    opt_g = torch.optim.RMSprop(self.generator.parameters(),lr=lr)
    opt_d = torch.optim.RMSprop(self.discriminator.parameters(),lr=lr)
    return[opt_g, opt_d],[]

  def plot_imgs(self):
    z = self.random_noise.type_as(self.generator.lin1.weight)
    images = self(z).cpu()
    print('epoch',self.current_epoch+1)
    fig = plt.figure()

    for i in range(images.size(0)):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(images.detach()[i,0,:,:],cmap='gray_r',interpolation='none')
      plt.title("generated images")
      plt.xticks([])
      plt.yticks([])
      plt.axis("off")
    plt.show()

  def on_train_epoch_end(self):
    self.plot_imgs()


if name == "main":
  
  data = MNISTDataModule()
  model = GAN()

  # training with
  trainer = pl.Trainer(max_epochs=50)
  trainer.fit(model,data)


