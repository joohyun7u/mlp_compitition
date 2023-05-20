import os
from os import listdir
from os.path import join, splitext
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose
import DnCNN
import argparse

# 랜덤 시드 고정
np.random.seed(42)

parser = argparse.ArgumentParser(description='Argparse')
# parser.add_argument('--epoch',          type=int,   default=80)
# parser.add_argument('--batch_size',     type=int,   default=128)
# parser.add_argument('--lr',             type=float, default=0.001)
parser.add_argument('--datasets_dir',   type=str,   default='/local_datasets/MLinP')
parser.add_argument('--csv',            type=str,   default='./best_dncnn_model1.pth')
parser.add_argument('--model',          type=str,   default='DnCNN')
parser.add_argument('--output_dir',     type=str,   default='../../output')
args = parser.parse_args()

# def load_img(filepath):
#     img = cv2.imread(filepath)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img

# # DnCNN 모델 정의
# class DnCNN(nn.Module):
#     def __init__(self, num_layers=17, num_channels=64):
#         super(DnCNN, self).__init__()
#         layers = [nn.Conv2d(3, num_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
#         for _ in range(num_layers - 2):
#             layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
#             layers.append(nn.BatchNorm2d(num_channels))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(num_channels, 3, kernel_size=3, padding=1))
#         self.dncnn = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.dncnn(x)
#         return out

if args.model == 'DnCNN':
    model = DnCNN.DnCNN()
else:
    model = DnCNN.DnCNN()
model.load_state_dict(torch.load(args.csv))
model.eval()

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# 데이터셋 경로
noisy_data_path = args.datasets_dir+'/test/scan'
output_path = args.output_dir

if not os.path.exists(output_path):
    os.makedirs(output_path)

class CustomDatasetTest(data.Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = DnCNN.load_img(self.noisy_image_paths[index])
        
        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path

test_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터셋 로드 및 전처리
print('전처리 실행')
noisy_dataset = CustomDatasetTest(noisy_data_path, transform=test_transform)

# 데이터 로더 설정
noisy_loader = DataLoader(noisy_dataset, batch_size=1, shuffle=False)

# 이미지 denoising 및 저장
for noisy_image, noisy_image_path in noisy_loader:
    noisy_image = noisy_image.to(device)
    noise = model(noisy_image)

    denoised_image = noisy_image - noise
    
    # denoised_image를 CPU로 이동하여 이미지 저장
    denoised_image = denoised_image.cpu().squeeze(0)
    denoised_image = torch.clamp(denoised_image, 0, 1)  # 이미지 값을 0과 1 사이로 클램핑
    denoised_image = transforms.ToPILImage()(denoised_image)

    # Save denoised image
    output_filename = noisy_image_path[0]
    denoised_filename = output_path + '/' + output_filename.split('/')[-1][:-4] + '.png'
    denoised_image.save(denoised_filename) 
    
    print(f'Saved denoised image: {denoised_filename}')


import os
import cv2
import csv
import numpy as np

folder_path = args.output_dir
out_num = 1
output_file = args.csv + 'output' + str(out_num) + '.csv'
while (os.path.isfile(folder_path + output_file)):
    out_num += 1
output_file = args.csv + 'output' + str(out_num) + '.csv'


# 폴더 내 이미지 파일 이름 목록을 가져오기
file_names = os.listdir(folder_path)
file_names.sort()

# CSV 파일을 작성하기 위해 오픈
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image File', 'Y Channel Value'])

    for file_name in file_names:
        # 이미지 로드
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        # 이미지를 YUV 색 공간으로 변환
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Y 채널 추출
        y_channel = image_yuv[:, :, 0]

        # Y 채널을 1차원 배열로 변환
        y_values = np.mean(y_channel.flatten())

        print(y_values)

        # 파일 이름과 Y 채널 값을 CSV 파일에 작성
        writer.writerow([file_name[:-4], y_values])

print('CSV file created successfully.')