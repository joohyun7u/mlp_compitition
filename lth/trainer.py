import argparse, time, importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

from utils import tool_box as T

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--epoch',          type=int,   default=80)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--val_rate',       type=float, default=0.1)
    parser.add_argument('--isCV',           type=float, default=None)
    parser.add_argument('--isSummary',      type=str,   default=False)
    parser.add_argument('--datasets_dir',   type=str,   default='/local_datasets/MLinP')
    parser.add_argument('--model',          type=str,   default=None)
    parser.add_argument('--model_save_dir', type=str,   default='./model_save/')
    parser.add_argument('--loss_save_dir',  type=str,   default='./loss_save/')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

    print(f"==================== TRAIN INFO =================== \
        \nrunning: {device}\
        \nModel: {args.model}\
        \nepoch: {args.epoch}\
        \nbatch: {args.batch_size}\
        \nlr_init: {args.lr}\
        \nSummary: {args.isSummary}\
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

    # 경로
    noisy_image_paths = dataset_dir+'train/scan/'
    clean_image_paths = dataset_dir+'train/clean/'

    # 로스함수 및 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 전처리
    train_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 데이터셋 설정
    dataset = T.CustomDataset(
        noisy_image_paths = noisy_image_paths, 
        clean_image_paths = clean_image_paths, 
        patch_size = 120,
        transform=train_transform
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

    
