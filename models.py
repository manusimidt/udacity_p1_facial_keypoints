# TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or
        # batch normalization) to avoid overfitting

        # data comes in as batch_size x 1 x 224 x 224
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        # batch_size x 32 x 224 x 224
        # self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # batch_size x 32 x 112 x 112

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        # batch_size x 64 x 109 x 109
        # self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # batch_size x 64 x 54 x 54

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        # batch_size = 128 x 52 x 52
        # self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # batch_size = 128 x 26 x 26

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        # batch_size = 256 x 25 x 25
        # self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # batch_size = 256 x 12 x 12

        self.fc1 = nn.Linear(in_features=256 * 12 * 12, out_features=512)
        self.fc1_drop = nn.Dropout(p=.4)
        # We want to get 68 keypoints (each having x and y coordinate) => 136 out_features
        self.fc2 = nn.Linear(in_features=512, out_features=136)

    def forward(self, x):
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        # input are 10 (batch_size) images with one color channel and with a resolution of 224x224
        # x: 10x1x224x224
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)

        x = F.leaky_relu(self.conv3(x))
        x = self.pool3(x)

        x = F.leaky_relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)  # 10x200704

        x = self.fc1(x)  # 10x2048
        x = F.leaky_relu(x)  # 10x2048
        x = self.fc1_drop(x)  # 10x2048

        x = self.fc2(x)  # 10x128
        # a modified x, having gone through all the layers of your model, should be returned
        return x
