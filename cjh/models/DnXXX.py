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
import DnCNN, resnet
from utils.param import param_check, seed_everything

# 이미지 로드 함수 정의
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 커스텀 데이터셋 클래스 정의
class CustomDataset(data.Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, patch_size = 128, transform=None):
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
        clean_image = load_img(self.clean_image_paths[index])

        H, W, _ = clean_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        clean_image = clean_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        # transform 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

# 모델 학습
def train(num_epochs):
    model.train()
    best_loss = 9999.0
    tem = 1
    for epoch in range(args.epoch):
        epoch_time = time.time()
        running_loss = 0.0
        for noisy_images, clean_images in train_loader:
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_images)
            if tem:
                tem=0
                print(outputs.size(), clean_images.size())
            loss = criterion(outputs, noisy_images-clean_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy_images.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f'Time: {time.time()-epoch_time:.0f}초 \tEpoch {epoch+1}/{num_epochs}, \tLoss: {epoch_loss:.4f}')

    # 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_file)
            print(f"{epoch+1}epoch 모델 저장 완료")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--epoch',          type=int,   default=80)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--datasets_dir',   type=str,   default='/local_datasets/MLinP')
    parser.add_argument('--csv',            type=str,   default='./best_dncnn_model1.pth')
    parser.add_argument('--model',          type=str,   default='DnCNN')
    parser.add_argument('--output_dir',     type=str,   default='~/output')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
    print(f"running: {device}, \tModel: {args.model} \tepoch: {args.epoch}, \tbatch: {args.batch_size}, \tlr: {args.lr}")

    # 랜덤 시드 고정
    np.random.seed(42)

    # 시작 시간 기록
    start_time = time.time()

    # 하이퍼파라미터 설정
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr

    dataset_dir = args.datasets_dir

    # 데이터셋 경로
    noisy_image_paths = dataset_dir+'/train/scan'
    clean_image_paths = dataset_dir+'/train/clean'

    # 모델 저장 위치 
    model_path = '../save/'
    model_num = 1
    model_file = 'best_dncnn_model' + str(model_num) + '.pth'
    while (os.path.isfile(model_path + model_file)):
        model_num += 1
        model_file = 'best_dncnn_model' + str(model_num) + '.pth'

    # 데이터셋 로드 및 전처리
    train_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 커스텀 데이터셋 인스턴스 생성
    train_dataset = CustomDataset(noisy_image_paths, clean_image_paths, transform=train_transform)

    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True, shuffle=True)

    m = args.model
    if m == 'DnCNN':
        model = DnCNN.DnCNN().to(device)
    elif m == 'ResNet18':
        model = resnet.resnet18().to(device)
    elif m == 'ResNet34':
        model = resnet.resnet34().to(device)
        print('이거')
    elif m == 'ResNet50':
        model = resnet.resnet50().to(device)
    elif m == 'ResNet101':
        model = resnet.resnet101().to(device)
    elif m == 'ResNet152':
        model = resnet.resnet152().to(device)
    else:
        model = DnCNN.DnCNN().to(device)
    param_check(model)
    param_check(model, True)
    print(summary(model, (3, 128, 128)))

    # 손실 함수와 최적화 알고리즘 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)

    # 모델 학습
    print("모델 학습 시작")
    train(args.epoch)

    # 종료 시간 기록
    end_time = time.time()

    # 소요 시간 계산
    training_time = end_time - start_time

    # 시, 분, 초로 변환
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)

    # 결과 출력
    print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")