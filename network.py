import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #Conv卷积核
        self.conv1 = torch.nn.Conv2d(3,40,[5,5],stride=[3, 3],padding=[0, 0])
        self.conv2 = torch.nn.Conv2d(40, 80, [3,3], stride=[2, 2], padding=[0, 0])
        self.conv3 = torch.nn.Conv2d(80, 120, [3,3], stride=[1, 1], padding=[1, 1])
        self.conv4 = torch.nn.Conv2d(120, 80, [3,3], stride=[1, 1], padding=[1, 1])
        #maxpooling池化层
        self.pooling1 = torch.nn.MaxPool2d(2, stride=[2, 2], padding=[0, 0])
        self.pooling2 = torch.nn.MaxPool2d(3, stride=[2, 2], padding=[1, 1])
        self.pooling3 = torch.nn.MaxPool2d(3, stride=[1, 1], padding=[1, 1])
        self.pooling4 = torch.nn.MaxPool2d(3, stride=[3, 3], padding=[1, 1])
        #fc全连接层
        self.fc1 = nn.Linear(2*2*80, 100)
        self.fc2 = nn.Linear(100, 33)

    def forward(self, x):
        x = self.conv1(x)  # conv
        x = F.relu(x)  # relu
        x = self.pooling1(x)  # maxpooling

        x = self.conv2(x)  # conv
        x = F.relu(x)  # relu
        x = self.pooling2(x)  # maxpooling

        x = self.conv3(x)  # conv
        x = F.relu(x)  # relu
        x = self.pooling3(x)  # maxpooling

        x = self.conv4(x)  # conv
        x = F.relu(x)  # relu
        x = self.pooling4(x)  # maxpooling

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        y = self.fc2(x)
        return y
