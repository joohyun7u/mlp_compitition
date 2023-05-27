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
    parser.add_argument('--version',        type=str)
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

    T.seed_everything(42)

    # 하이퍼파라미터 설정
    version = args.version
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
    noise_image_paths = dataset_dir+'train/residuals/'
    clean_image_paths = dataset_dir+'train/clean/'
    noisy_image_paths = dataset_dir+'train/scan/'

    # 로스함수 및 옵티마이저 설정

    criterion = VGGPerceptualLoss(model="vgg16").to(device)
    #criterion = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)


    criterion_name = str(criterion).split('\n')[0]
    optimizer_name = str(optimizer).split('\n')[0]
    print(f"========================= TRAIN INFO =========================== \
        \nRunning on: {device}\
        \nModel: {model_name}\
        \nVersion: {version}\
        \nEpoch: {num_epochs}\
        \nBatch: {batch_size}\
        \nLr_init: {learning_rate}\
        \nCriterion: {criterion_name}\
        \nOptimizer: {optimizer_name}\
        \n================================================================")
    torchsummary.summary(model,input_size=(3,512,512))
    T.param_check(model=model)


    # 학습
    trainer = T.Trainer(
        model = model,
        version = version,
        model_name = model_name,
        num_epochs = num_epochs,
        train_data_loader = train_loader,
        valid_data_loader = valid_loader,
        optimizer = optimizer,
        criterion = criterion,
        scheduler = scheduler,
        model_save_dir = model_save_path,
        loss_save_dir = loss_save_dir
    )
    trainer.train()

    
