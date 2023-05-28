import argparse, importlib

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

from utils import tool_box as T
from utils.loss_vgg import VGGPerceptualLoss
from utils.loss_charbon import CharbonnierLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--version',        type=str)
    parser.add_argument('--total_iteration',type=int)
    parser.add_argument('--batch_size',     type=int)
    parser.add_argument('--learning_rate',  type=float)
    parser.add_argument('--val_rate',       type=float)
    parser.add_argument('--datasets_dir',   type=str)
    parser.add_argument('--model_name',     type=str)
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--validation_output_dir', type=str)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

    T.seed_everything(42)

    # 하이퍼파라미터 설정
    version = args.version
    total_iteration = args.total_iteration
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    val_rate = args.val_rate
    dataset_dir = args.datasets_dir
    model_name = args.model_name
    model_save_dir = args.model_save_dir
    validation_output_dir = args.validation_output_dir

    # 모델 로딩
    model = importlib.import_module('.' + model_name, '.models').model

    # 경로
    clean_image_paths = dataset_dir+'train/clean/'
    noisy_image_paths = dataset_dir+'train/scan/'


    # 로스함수 및 옵티마이저 설정

    criterion = CharbonnierLoss(eps = 1e-3)
    #criterion = VGGPerceptualLoss(model="vgg16").to(device)
    #criterion = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    scheduler = MultiStepLR(optimizer,
                            [250000, 400000, 450000, 475000, 500000],
                            0.5, 
                            verbose = False
                )
    

    # 전처리
    noisy_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    clean_transform = Compose([
        ToTensor(),
    ])

    # 데이터셋 설정
    train_set = T.TrainDataset(
        noisy_image_paths = noisy_image_paths, 
        clean_image_paths = clean_image_paths, 
        patch_size = 64,
        noisy_transform=noisy_transform,
        clean_transform=clean_transform
    )

    train_size = int(len(train_set)*(1-val_rate))
    val_size = len(train_set) - train_size
    train_set, valid_set = random_split(train_set,[train_size,val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4)

    # 학습
    trainer = T.Trainer(
        model = model,
        version = version,
        model_name = model_name,
        total_iteraion = total_iteration,
        train_data_loader = train_loader,
        valid_data_loader = valid_loader,
        validation_output_dir = validation_output_dir,
        validation_checkpoint = 10000,
        optimizer = optimizer,
        criterion = criterion,
        scheduler = scheduler,
        model_save_dir = model_save_dir,
    )

    trainer.train_info()
    trainer.train()

    
