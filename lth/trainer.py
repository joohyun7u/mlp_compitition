import argparse, importlib

import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torchsummary

from utils import tool_box as T
from utils.custom_transforms import NoiseReconstruct
from utils.vgg_perceptual_loss import VGGPerceptualLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--epoch',          type=int)
    parser.add_argument('--batch_size',     type=int)
    parser.add_argument('--lr',             type=float)
    parser.add_argument('--val_rate',       type=float)
    parser.add_argument('--isCV',           type=float)
    parser.add_argument('--datasets_dir',   type=str)
    parser.add_argument('--model',          type=str)
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--loss_save_dir',  type=str)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

    print(f"==================== TRAIN INFO =================== \
        \nrunning: {device}\
        \nModel: {args.model}\
        \nepoch: {args.epoch}\
        \nbatch: {args.batch_size}\
        \nlr_init: {args.lr}\
        \n===================================================")

    T.seed_everything(42)

    # 하이퍼파라미터 설정
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    val_rate = args.val_rate
    isCrossVal = args.isCV
    dataset_dir = args.datasets_dir
    model_name = args.model
    model_save_path = args.model_save_dir
    loss_save_dir = args.loss_save_dir

    # 모델 로딩
    model = importlib.import_module('.' + model_name, '.models').model
    torchsummary.summary(model,input_size=(3,512,512))
    T.param_check(model=model)

    # 경로
    noise_image_paths = dataset_dir+'train/residuals/'
    clean_image_paths = dataset_dir+'train/clean/'
    noisy_image_paths = dataset_dir+'train/scan/'

    # 로스함수 및 옵티마이저 설정

    criterion = VGGPerceptualLoss(model="vgg16").to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2e5, gamma=0.5)

    # 전처리
    noisy_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    noise_transform = Compose([
        ToTensor(),
        NoiseReconstruct()
    ])

    # 데이터셋 설정
    dataset = T.CustomDataset(
        noisy_image_paths = noisy_image_paths, 
        noise_image_paths = noise_image_paths, 
        patch_size = 128,
        noisy_transform=noisy_transform,
        noise_transform=noise_transform
    )

    train_size = int(len(dataset)*(1-val_rate))
    val_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    # 학습
    trainer = T.Trainer(
        model = model,
        model_name = model_name,
        num_epochs = num_epochs,
        train_data_loader = train_loader,
        valid_data_loader = valid_loader,
        optimizer = optimizer,
        criterion = criterion,
        model_save_dir = model_save_path,
        loss_save_dir = loss_save_dir
    )
    trainer.train()

    
