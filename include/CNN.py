transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(
    root="/content/sample_data",
    transform = transform,
    download=True,
    train = True
)

test_data = datasets.FashionMNIST(
    root="/content/sample_data",
    transform = transform,
    download=True,
    train = False
)

train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10,shuffle=True)

for images,labels in train_loader:
  break

print(images.shape)

# Weird code that plots the images 
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
'''
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''

# Neural Network class 
class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(1,6,3,1)
    self.conv2 = nn.Conv2d(6,16,3,1)
    #### TODO calculation here!!
    self.fc1 = nn.Linear(5*5*16, 100)
    ###
    self.fc2 = nn.Linear(100,84)
    self.fc3 = nn.Linear(84,10)

  def forward(self,X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2)
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2)
    X = X.view(-1, 5*5*16)
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return F.log_softmax(X,dim=1)



torch.manual_seed(101)

Model = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(),lr=0.001)

epochs = 5



for i in range(epochs):

    for b,(X_train,y_train) in enumerate(train_loader):
      y_pred = Model(X_train)
      loss = criterion(y_pred,y_train)
      predicted = torch.max(y_pred.data,1)[1]
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(f"Epoch:{i}, Loss:{loss.item()}")
     

with torch.no_grad():
  pass
  





