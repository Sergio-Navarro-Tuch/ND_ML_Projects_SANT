import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt    #Model for plots generation


import time   #module for timing the progress and actions (used to optimize timing in redesign)
import torch  #importing pytorch module
from torch import nn  #from Pytorch import Neural Networks
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict

import json

import PIL
from PIL import Image

import argparse

import numpy as np
import os, random

import FunctionsAidAI as FAID  #import functions from aiding file

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

plt.rcParams["figure.figsize"] = (15,10)
plt.tight_layout()

plt.subplot(2,2,1)
plt.plot(x, y)
plt.title('Plot 1')

plt.subplot(2,2,2)
plt.scatter(x, y)
plt.title('Plot 2')

plt.subplot(2,2,3)
plt.plot(x, y)
plt.title('Plot 3')

plt.subplot(2,2,4)
plt.scatter(x, y)
plt.title('Plot 4')
plt.show()
