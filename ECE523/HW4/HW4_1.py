# HW 4 problem 1
# Samuel Freitas
# ECE 523 

# reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import numpy as np 
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
from tabulate import tabulate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available() :
    print('Using GPU:', torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print('Using the cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5,stride = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 5, stride = 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn_func.relu(self.conv1(x)))
        x = self.pool(nn_func.relu(self.conv2(x)))
        # x = self.pool(nn_func.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn_func.relu(self.fc1(x))
        x = nn_func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# set the network and point it to the GPU
net = Net()
net.to(device)

# save a blank network
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# loss funtion selection
criterion = nn.CrossEntropyLoss()

# create some variables for the loop
# weight_decay is L2 norm
L2_weight_penalty_list = [0,1e-5,0,1e-5]
number_epochs_list = [2,2,10,10]
results = []

for i in range(4):

    L2_weight_penalty = L2_weight_penalty_list[i]
    learning_rate = 0.001
    momentum_rate = 0.9
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum_rate,weight_decay=L2_weight_penalty)

    net.load_state_dict(torch.load(PATH))

    number_epochs = number_epochs_list[i]
    for epoch in range(number_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 12000 == 11999:    # print every 12000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    
                end_running_loss = running_loss
                running_loss = 0.0

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.cuda(), labels.cuda()

    # net = Net()
    # net.to(device)
    # PATH = './cifar_net.pth'
    # net.load_state_dict(torch.load(PATH))
    outputs = net(images).to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc_correct = 100 * correct / total

    print('Accuracy:', acc_correct, '% - L2 weight penalty:',L2_weight_penalty, ' - Epochs:', number_epochs)

    this_result = [number_epochs,L2_weight_penalty,acc_correct,end_running_loss]
    results.append(this_result)

print(tabulate(results, headers=['Num epochs','L2 penalty','Accuracy','loss']))