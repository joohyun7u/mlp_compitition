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
import models.DnCNN as DnCNN, models.ResNet as ResNet, models.RFDN as RFDN
from utils.param import param_check, seed_everything

model_list = ['DnCNN', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101','ResNet152', 'RFDN']
models = {'DnCNN': DnCNN.DnCNN(), 
          'ResNet18': ResNet.ResNet18(), 
          'ResNet34': ResNet.ResNet34(), 
          'ResNet50': ResNet.ResNet50(), 
          'ResNet101': ResNet.ResNet101(), 
          'ResNet152': ResNet.ResNet152(), 
          'RFDN': RFDN.RFDN()
          }

device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
for m in model_list:
    if m == 'DnCNN':
        model = DnCNN.DnCNN().to(device)
    elif m == 'ResNet18':
        model = ResNet.ResNet18().to(device)
    elif m == 'ResNet34':
        model = ResNet.ResNet34().to(device)
    elif m == 'ResNet50':
        model = ResNet.ResNet50().to(device)
    elif m == 'ResNet101':
        model = ResNet.ResNet101().to(device)
    elif m == 'ResNet152':
        model = ResNet.ResNet152().to(device)
    elif m == 'RFDN':
        model = RFDN.RFDN().to(device)
   
    print('\n\n',m,' 모델은 다음과 같다.')
    param_check(model)
    param_check(model, True)
print(summary(model, (3, 128, 128)))