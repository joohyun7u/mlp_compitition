import argparse, cv2, matplotlib.pyplot as plt,numpy as np,os
from os.path import join

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils.tool_box as T


if __name__ == '__main__':
    train_src_dir = '/local_datasets/MLinP/train/scan'
    test_src_dir = '/local_datasets/MLinP/test/scan'
    
    train_dst_dir = '/local_datasets/MLinP/train/bilateral_scan'
    test_dst_dir = '/local_datasets/MLinP/test/bilateral_scan'

    if not os.path.exists(train_dst_dir):
        os.makedirs(train_dst_dir)

    if not os.path.exists(test_dst_dir):
        os.makedirs(test_dst_dir)


    train_src_dataset = T.TestDatastLoader(
        noisy_image_paths = train_src_dir,
        noisy_transform = transforms.ToTensor()
    )

    test_src_dataset = T.TestDatastLoader(
        noisy_image_paths = test_src_dir,
        noisy_transform = transforms.ToTensor()
    )
    
    train_src_loader = DataLoader(train_src_dataset, batch_size=1, shuffle=False)
    test_src_loader = DataLoader(test_src_dataset, batch_size=1, shuffle=False)

    for i, (noisy_image, noisy_image_path) in enumerate(train_src_loader):

        filename = noisy_image_path[0].split('/')[-1]

        noisy_image = noisy_image.squeeze(0)
        noisy_image = torch.clamp(noisy_image,0,1)
        noisy_image = transforms.ToPILImage()(noisy_image)
        noisy_image = np.array(noisy_image)

        output = cv2.bilateralFilter(noisy_image, 5, 75, 75)

        output_filename = noisy_image_path[0]
        output_filename = join(train_dst_dir, output_filename.replace('\\', '/').split('/')[-1][:-4] + '.png')

        output = transforms.ToPILImage()(output)
        output.save(output_filename)

        print(f'filesave: {output_filename}') 

    print("=================training_set_bilateral_filter complete===================")

    for i, (noisy_image, noisy_image_path) in enumerate(test_src_loader):

        filename = noisy_image_path[0].split('/')[-1]

        noisy_image = noisy_image.squeeze(0)
        noisy_image = torch.clamp(noisy_image,0,1)
        noisy_image = transforms.ToPILImage()(noisy_image)
        noisy_image = np.array(noisy_image)

        output = cv2.bilateralFilter(noisy_image, 5, 75, 75)

        output_filename = noisy_image_path[0]
        output_filename = join(test_dst_dir, output_filename.replace('\\', '/').split('/')[-1][:-4] + '.png')

        output = transforms.ToPILImage()(output)
        output.save(output_filename)

        print(f'filesave: {output_filename}')