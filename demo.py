import csv

import torch
import imageio
import numpy as np

from collections import namedtuple
from torch import nn
from PIL import Image
from collections import OrderedDict
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from torchvision.models import ResNet101_Weights

# seq_model1 = nn.Sequential(OrderedDict([
#     ('hidden_linear', nn.Linear(1, 13)),
#     ('hidden_activation', nn.Tanh()),
#     ('output_linear', nn.Linear(13, 1))
# ]))
#
# data_path = '../data/cifar10/'
# cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
# cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.act1 = nn.Tanh()
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
#         self.act2 = nn.Tanh()
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(8 * 8 * 8, 32)
#         self.act3 = nn.Tanh()
#         self.fc2 = nn.Linear(32, 2)
#
#     def forward(self, x):
#         out = self.pool1(self.act1(self.conv1(x)))
#         out = self.pool2(self.act2(self.conv2(out)))
#         out = out.view(-1, 8 * 8 * 8)
#         out = self.act3(self.fc1(out))
#         out = self.fc2(out)
#         return out
#
#
# model = Net()
color = namedtuple('Color', 'red green blue')
color1 = color(1, 2, 3)
print(color1)