import random, numpy as np, cv2, time, csv
import os
from os import listdir
from os.path import join

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

##################### Let's make some noise ######################################

class CleanDatastLoader(data.Dataset):
    def __init__(self, clean_image_paths, clean_transform):
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]
        self.clean_transform = clean_transform

    def __len__(self):
        return len(self.clean_image_paths)

    def __getitem__(self, index):
        
        clean_image_path = self.clean_image_paths[index]
        clean_image = load_img(self.clean_image_paths[index])
        
        clean_image = self.clean_transform(clean_image)

        return clean_image, clean_image_path

######################################################################

class Noiser():
    def __init__(self, model, model_name, image_size, window_size, model_save_dir ,model_pth_name, clean_data_loader, image_output_dir):
        self.model = model
        self.model_name = model_name
        self.image_size = image_size
        self.window_size = window_size
        self.model_save_dir = model_save_dir
        self.model_pth_name = model_pth_name
        self.clean_data_loader = clean_data_loader
        self.image_output_dir = image_output_dir

        # 모델 파라미터 로딩
        pth_filename = join(model_save_dir, model_pth_name + '.pth')
        print(f"pth_filename: {pth_filename}")
        model.load_state_dict(torch.load(pth_filename))
        model.eval()

        if not os.path.exists(self.image_output_dir):
            os.makedirs(self.image_output_dir)

    def make_some_noise(self):
        self.model.eval()
        for iter, (clean_image, clean_image_path) in enumerate(self.clean_data_loader):
            
            clean_image = clean_image.to(device)

            with torch.no_grad():
                # pad input image to be a multiple of window_size
                img_lq = clean_image
                h_old, w_old = self.image_size
                h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
                w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = self.model(img_lq)
                noised_image = output[..., :h_old, :w_old]
                noised_image = torch.clamp(noised_image,0,1)
            
            output_filename = clean_image_path[0]

            noised_image = transforms.ToPILImage()(noised_image.squeeze(0))
            noised_filename = join(self.image_output_dir, output_filename.replace('\\','/').split('/')[-1][:-4] + '.png')
            noised_image.save(noised_filename)

            print(f'Saved noised image: {noised_filename}')

############## 유틸리티 ################
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img