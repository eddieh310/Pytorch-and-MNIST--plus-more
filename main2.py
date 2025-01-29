import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# for this project I chose to use the pytorch library
# much of the code here is refrenced from the offical pytorch documentation
# this can be found here: https://pytorch.org/docs/stable/nn.html#convolution-layers
# and here https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

transform = transforms.Compose([transforms.ToTensor()]) # transformer that will be used to convert data to tensor
# these next two lines load in MNIST dataset from torchvision library
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform) 
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

learning_rate = 0.01 # i had to change learning rate to .01, as the handout stated to use 10 but this 
# led to very low accuracy
batch_size = 32 # batch size of 32 as per the handout
epochs = 5 # start with 5 epochs, may change later

# here we create the dataloaders, which are just iterables that make it easy to work with our datasets
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # shuffle data so its different every time
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# next step is to define a class for the CNN model, as this is standard practice when working with pytorch

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        # here we set up the CNN 
        # channels is 1 because its greyscale, 
        # 32 out channels bc we have 32 filters
        # kernel size 3x3 as per handout
        # stride of 1 as per handout
        # handout stated to use valid padding

        # i follow the exact steps from the handout, first convolution layer, relu activation
        # then pooling, second convolution layer, 3 linear fully connected layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(32 * 5 * 5, 100) # 32*5*5 because of the pooling 
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, num_classes)
        # at this point we have initalized everything we need

    def forward(self, x):
        # this method simulated a forward pass of some data , x, through out network
        # we use the methods we defined in the init()
        # again, we follow exact order as per the handout
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # here i ran into an issue where the shapes mismatched
        # this line reshapes the tensor into a 2d tensor
        # we need the tensor to be 2d for teh linear layer
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        # the handout stated to use sigmoid activation function for the fully connected layer
        # i could not achieve an accuracy that made any sense with this approach, so i used relu here instad and received much better results
        return x

# initialize model
model = CNNModel(num_classes=10)
mse_function = nn.MSELoss()# use mse loss as per handout 
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # momentum is a further optimizatoin that pytorch allows
# this optimizer is the sgd optimizer, with learning rate of .01

# begin training loop
# the main layout of this training loop was taken from:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# however i had to make some variations for this specific case
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad() # reset gradients
        outputs = model(images) # pass inputs through model
        # i ran into an issue when using MSE loss where the tensors shapes were mismatched
        # i found that using this one_not method i could manipulate it as needed so the labels are same shape
        labels_one_hot = F.one_hot(labels, num_classes=10).float()
        # compute the loss
        loss = mse_function(outputs, labels_one_hot)
        loss.backward() # backprop
        optimizer.step() # apply weight updates
        running_loss += loss.item() # accumulate loss
    print("Epoch",epoch,"Loss:", running_loss / len(train_loader))


# at this point, training is complete, so we will evaluate the model and compute accuracy

model.eval() # set model to evaluation mode
correct = 0
total = 0 # counters
with torch.no_grad(): # dont need gradient caluclaton during testing
    for images, labels in train_loader:
        outputs = model(images) # pass inputs through model
        predicted = torch.argmax(outputs, dim=1) # get max output, which corresponds to predicted class
        correct += (predicted == labels).sum().item() # update correct counter if the predicted value = actual label
        total += labels.size(0) # add to total to keep track of how many weve tested
train_accuracy = correct / total # calculate accuracy

# now we repeat same steps but using the testing data

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
test_accuracy = correct / total

print("Train Accuracy:", train_accuracy,"Test Accuracy:", test_accuracy)
