## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        # Conv Layer
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3) 
        self.conv3 = nn.Conv2d(64, 128, 3) 
        self.conv4 = nn.Conv2d(128, 256, 2) 

        # BatchNorm2d Layer
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_bn = nn.BatchNorm2d(256)

        # Maxpooling Layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)

        # Dropout
        self.conv1_drop = nn.Dropout(0.1)
        self.conv2_drop = nn.Dropout(0.2)
        self.conv3_drop = nn.Dropout(0.3)
        self.conv4_drop = nn.Dropout(0.4)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2_drop = nn.Dropout(0.5)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # First Conv layer: (1, 224, 224) -> (32, 220, 220) -> (32, 110, 110)
        x = self.pool(F.leaky_relu(self.conv1_bn(self.conv1(x))))
        x = self.conv1_drop(x)

        # Second Conv layer: (32, 110, 110) -> (64, 108, 108) -> (64, 54, 54)
        x = self.pool(F.leaky_relu(self.conv2_bn(self.conv2(x))))
        x = self.conv2_drop(x)

        # Third Conv layer: (64, 54, 54) -> (128, 52, 52) -> (128, 26, 26)
        x = self.pool(F.leaky_relu(self.conv3_bn(self.conv3(x))))
        x = self.conv3_drop(x)

        # Forth Conv layer: (256, 26, 26) -> (256, 25, 25) -> (256, 12, 12)
        x = self.pool(F.leaky_relu(self.conv4_bn(self.conv4(x))))
        x = self.conv4_drop(x)

        # Flatten Layer: (256, 12, 12) -> (256*12*12)
        x = x.view(x.size(0), -1)

        # First Fc: (256*12*12) -> (1000) 
        x = self.fc1_drop(F.leaky_relu(self.fc1(x)))

        # Second Fc:  (1000) -> (1000)
        x = self.fc2_drop(F.leaky_relu(self.fc2(x)))

        # Output Fc: (1000) -> (136)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
