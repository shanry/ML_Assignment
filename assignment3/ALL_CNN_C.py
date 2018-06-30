import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import time

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

size_batch = 256

def load_data():

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False)
    train_mean = trainset.train_data.mean(axis=(0, 1, 2)) / 255  # [0.49139968  0.48215841  0.44653091]
    train_std = trainset.train_data.std(axis=(0, 1, 2)) / 255  # [0.24703223  0.24348513  0.26158784]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(train_mean, train_std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=size_batch,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=size_batch,
                                             shuffle=False)
    print('Finished Loading Data')
    return trainloader, testloader



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)

        self.conv7 = nn.Conv2d(192, 192, 3)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)

        self.avepool = nn.AvgPool2d(6)

    def forward(self, x):
        x = self.drop1(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.drop2(self.conv3(x)))

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.drop3(self.conv6(x)))

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))

        x = self.avepool(x)

        x = x.view(-1, 10)

        return x



mynet = Net().cuda()


def train(epoch=350, lr=0.001, data_train=None, net=mynet):
    # dataiter = iter(data_train)
    print("epoch =", epoch, "lr =", lr, "batchsize= ", size_batch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)
    for ep in range(epoch):  # loop over the dataset multiple times
        net.train()
        net = net.cuda()
        running_loss = 0.0
        scheduler.step()
        for i, data in enumerate(data_train, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (ep + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        if ep % 5 == 4:
            net.eval()
            test(testloader)

    print('Finished Training')


def test(data_test=None,net=mynet):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_test:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in data_test:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    start = time.time()
    trainloader, testloader = load_data()  #load data
    print("--- %.3f seconds ---" % (time.time() - start))
    test(data_test=testloader)
    print("--- %.3f  seconds ---" % (time.time() - start))
    train(epoch=350, lr=0.001, data_train=trainloader)
    print("--- %.3f  seconds ---" % (time.time() - start))
    test(data_test=testloader)
    print("--- %.3f  seconds ---" % (time.time() - start))
