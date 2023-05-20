import random
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from os.path import join
from os import listdir
from torchsummary import summary
import time
import argparse
import DnCNN
from utils.param import param_check, seed_everything

model_list = ['DnCNN']

device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
for m in model_list:
    if m == 'DnCNN':
        model = DnCNN.DnCNN().to(device)
   
    print(m,'모델은 다음과 같다.')
    param_check(model)
    param_check(model, True)