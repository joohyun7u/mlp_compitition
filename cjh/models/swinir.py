import random
import argparse
import cv2
import glob
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as tf
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision.transforms.functional as F
from os.path import join
from os import listdir
from torchsummary import summary
import time
from collections import OrderedDict
import os
import torch
import requests
from models.network_swinir import SwinIR as net
from models.swin_transformer_v2 import SwinTransformerV2 as net2
from models.kbnet_s_arch import KBNet_s
from models.kbnet_l_arch import KBNet_l
from models.restormer_arch import Restormer
from utils import util_calculate_psnr_ssim as util
from utils.param import param_check, seed_everything
import utils.vgg_loss, utils.vgg_perceptual_loss
from PIL import Image
import matplotlib.pyplot as plt
import gc
from math import log10
from torch.autograd import Variable
from utils.lr_scheduler import CosineAnnealingRestartCyclicLR
gc.collect()
torch.cuda.empty_cache()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss

# -1 ~ 1사이의 값을 0~1사이로 만들어준다
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# 이미지 시각화 함수
def show_images(real_a, real_b, fake_b):
    plt.figure(figsize=(30,90))
    plt.subplot(131)
    plt.imshow(real_a.cpu().data.numpy().transpose(1,2,0))
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(132)
    plt.imshow(real_b.cpu().data.numpy().transpose(1,2,0))
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(133)
    plt.imshow(fake_b.cpu().data.numpy().transpose(1,2,0))
    plt.xticks([])
    plt.yticks([])
    
    plt.show()


# 이미지 로드 함수 정의
def load_img(filepath, noise=0):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# 커스텀 데이터셋 클래스 정의
class CustomDataset(data.Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, patch_size = 128, transform=None, clean_transform=None, noisy_train = False):
        # super(Dataset, self).__init__() # 초기화 상속
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]# a는 건물 사진
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]# b는 Segmentation Mask
        self.transform = transform
        self.clean_transform = clean_transform
        self.patch_size = patch_size
        self.noisy_train = noisy_train
        # self.image_filenames = [x for x in os.listdir(self.a_path)] # a 폴더에 있는 파일 목록

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index], args.noise)
        clean_image = load_img(self.clean_image_paths[index])
        if self.noisy_train:
            clean_image = noisy_image - clean_image

        H, W, _ = clean_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        clean_image = clean_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        # transform 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
        if self.clean_transform:
            clean_image = self.clean_transform(clean_image)

        # 이미지 랜덤 수평 플립
        if torch.rand(1) < 0.5:
            noisy_image = F.hflip(noisy_image)
            clean_image = F.hflip(clean_image)
        
        # 이미지 랜덤 수직 플립
        if torch.rand(1) < 0.5:
            noisy_image = F.vflip(noisy_image)
            clean_image = F.vflip(clean_image)
        
        return noisy_image, clean_image

class loss_save():
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_mae = []
        self.val_mae = []

    def add(self, model, epoch, train_loss, val_loss, save_dir, train_mae=None, val_mae=None):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        if train_mae:
            self.train_mae.append(train_mae)
        if val_mae:
            self.val_mae.append(val_mae)
        torch.save(
            {
                "model": model,
                "epoch": self.epochs,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "train_mae": self.train_mae,
                "val_mae": self.val_mae
            }, save_dir
    )

def tensor_to_yuv(images):
    i =[]
    for image in images:
        image = torch.clamp(image, 0, 1)  # 이미지 값을 0과 1 사이로 클램핑
        image = tf.ToPILImage()(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        i.append(image)
    return i

def cal_mae_loss(ps,ts, y=False):
    abss = 0.0
    for p, t in zip(ps,ts):
        if not y:
            p = p[:, :, 0]
            t = t[:, :, 0]
        abss += abs(np.mean(p.flatten()) - np.mean(t.flatten()))
    return abss

# 모델 학습
def train(num_epochs, noise = True, save_val=True, model_type=False, load_epoch=0):
    best_loss = 9999.0
    best_val_loss = 9999.0
    tem = 1
    total_iter = len(train_loader)
    loss_pth = loss_save()
    if not model_type:
        print(0)
        for epoch in range(load_epoch, args.epoch):
            model.train()
            epoch_time = time.time()
            running_loss = 0.0
            tot_mae = 0.0
            for iter, (noisy_images, clean_images) in enumerate(train_loader):
                noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
                optimizer.zero_grad()
                outputs = model(noisy_images) # .tolist()
                if noise:
                    loss = criterion(outputs, noisy_images-clean_images)
                else:
                    loss = criterion(outputs, clean_images)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item() * noisy_images.size(0)
                cal_mae = cal_mae_loss(tensor_to_yuv(outputs),tensor_to_yuv(clean_images))
                tot_mae += cal_mae
                if (iter+1) % int(total_iter/8) == 0:
                    print(f"\t[{iter+1}/{total_iter}] \tlr: {optimizer.param_groups[0]['lr']} \tTrain_Loss: {loss.item():.4f}\tMAE: {cal_mae/noisy_images.size(0):.4f}")
                
            epoch_loss = running_loss / len(train_dataset)
            val_loss , val_mae = val(noise)
            loss_pth.add(args.model,epoch,epoch_loss,val_loss,loss_file,tot_mae/len(train_dataset),val_mae)
            print(f'Epoch {epoch+1}/{num_epochs} \tTime: {time.time()-epoch_time:.0f}초 \tTrain_Loss: {epoch_loss:.4f} \tVal_Loss: {val_loss:.4f} \tMAE: {tot_mae/len(train_dataset):.5f} \tVAL_MAE: {val_mae}')
            torch.save(model.state_dict(), './_save/'+args.load_pth_name)
        # 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
            if save_val:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_file)
                    print(f"\t{epoch+1}epoch 모델 저장 완료")
            else:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), model_file)
                    print(f"\t{epoch+1}epoch 모델 저장 완료")


def val(noise = True):
    model.eval()
    running_loss = 0.0
    tot_mae = 0.0
    with torch.no_grad():
        for noisy_images, clean_images in val_loader:
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            outputs = model(noisy_images)
            if noise:
                loss = criterion(outputs, noisy_images-clean_images)
            else:
                loss = criterion(outputs, clean_images)
            running_loss += loss.item() * noisy_images.size(0)
            tot_mae += cal_mae_loss(tensor_to_yuv(outputs),tensor_to_yuv(clean_images))
        epoch_loss = running_loss / len(val_dataset)
    return epoch_loss, tot_mae/len(val_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--epoch',          type=int,   default=80)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--val',            type=float, default=0.1)
    parser.add_argument('--cv',             type=float, default=None)
    loss_list =  ['MSELoss', 'vgg_loss', 'vgg_perceptual_loss']
    parser.add_argument('--loss',           type=int,   default=1)
    parser.add_argument('--train_img_size', type=int,   default=128)
    parser.add_argument('--noise_train',    type=str,   default='False')
    parser.add_argument('--val_best_save',  type=str,   default='true')
    parser.add_argument('--summary',        type=str,   default=False)
    parser.add_argument('--datasets_dir',   type=str,   default='/local_datasets/MLinP')
    parser.add_argument('--csv',            type=str,   default='./best_dncnn_model1.pth')
    parser.add_argument('--model',          type=str,   default='DnCNN')
    parser.add_argument('--output_dir',     type=str,   default='~/output')
    parser.add_argument('--noise',          type=int,   default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--load_pth_name',  type=str,   default='temp', help='다시 시작할 pth')
    parser.add_argument('--load_epoch',     type=int,   default=0, help='다시 시작할 epoch')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
    # 랜덤 시드 고정
    seed_everything(42)
    bools = {'true' : True, 'True' : True, 'TRUE' : True, 'false' : False, 'False' : False, 'FALSE' : False}

    # 시작 시간 기록
    start_time = time.time()

    # 하이퍼파라미터 설정
    if bools[args.noise_train]:
        end_pth = '.pth'
        result_noise = True
    else:
        end_pth = '_clean.pth'
        result_noise = False
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr

    dataset_dir = args.datasets_dir

    # 데이터셋 경로
    noisy_image_paths = dataset_dir+'/train/scan'
    clean_image_paths = dataset_dir+'/train/clean'
    print('진행중?')
    if args.model == 'swinir':
        model = net(upscale=1, in_chans=3, img_size=args.train_img_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv').to(device)
    elif args.model == 'swinirv2':
        model = net(upscale=1, in_chans=3, img_size=args.train_img_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv').to(device)
    elif args.model =='KBNet':
        model = KBNet_s(middle_blk_num=2, enc_blk_nums=[1, 2, 2], dec_blk_nums=[1, 1, 2],lightweight=True).to(device)
    elif args.model == 'Restormer':
        model = Restormer(dim = 30, num_blocks = [2,3,6,8], num_refinement_blocks = 4, heads = [1,2,4,8]).to(device)
    
    # model = net(upscale=1, in_chans=3, img_size=args.train_img_size, window_size=8,
    #                 img_range=1., embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    #                 ).to(device)
    # img_E = utils_model.test_mode(model, img_L, mode=2, min_size=480, sf=sf)  # use this to avoid 'out of memery' issue.

    if args.load_epoch:
        model.load_state_dict(torch.load('./_save/'+args.load_pth_name))
        print(f'epoch 재개완료 {args.load_epoch+1} 부터 시작')
    
    model_type = False
    param_check(model)
    param_check(model, True)
    if args.summary == 'True' or args.summary == 'true':
        print(summary(model, (3, 128, 128)))

    border = 0
    window_size = 8

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

    if args.model =='KBNet' or args.model =='Restormer':
        args.lr = 3e-4
        if args.loss == 2:
            vgg_model = 'vgg16'
            args.loss = 'VGGPerceptualLoss'
            print('! vgg model :' , vgg_model)
            criterion = utils.vgg_perceptual_loss.VGGPerceptualLoss(model=vgg_model).to(device)
        elif args.loss == 3:
            args.loss = 'L1Loss'
            criterion = nn.L1Loss()
        else:
            args.loss = 'CharbonnierLoss'
            criterion = CharbonnierLoss(1e-3)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingRestartCyclicLR(optimizer,[92000, 208000,408000,808000],[1,1,0.1,0.05],[0.0003,0.000001,0.000001,0.000001])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2e5, gamma=0.5)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 'min')
        if args.loss == 2:
            vgg_model = 'vgg16'
            args.loss = 'VGGPerceptualLoss'
            print('! vgg model :' , vgg_model)
            criterion = utils.vgg_perceptual_loss.VGGPerceptualLoss(model=vgg_model).to(device)
        elif args.loss == 3:
            args.loss = 'L1Loss'
            criterion = nn.L1Loss()
        else:
            criterion = CharbonnierLoss(1e-3)
        args.loss = 'CharbonnierLoss'
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                [250000, 400000, 550000, 650000, 700000],
                                0.5, 
                                verbose = False
                    )
   
    print("lr: ", optimizer.param_groups[0]['lr'])
    

     # 데이터셋 로드 및 전처리
    train_transform = Compose([
        # BilateralBlur(args.train_img_size),
        # tf.ToPILImage(),
        # tf.RandAugment(),
        # ToTensor(),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_clean_transform = Compose([
        # BilateralBlur(args.train_img_size),
        ToTensor()
    ])

    val_transform = Compose([
        # BilateralBlur(args.train_img_size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 커스텀 데이터셋 인스턴스 생성
    dataset = CustomDataset(noisy_image_paths, clean_image_paths, args.train_img_size, transform=train_transform, clean_transform = train_clean_transform, noisy_train = bools[args.noise_train])
    
    # val 분할
    train_size = int(len(dataset)*(1-args.val))
    val_size = int(len(dataset)*(args.val))
    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])
    print(f"Train Size : {len(train_dataset)} \tValidattion Size : {len(val_dataset)}")
    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=6, 
                              persistent_workers=True, 
                              drop_last=True,
                              pin_memory=True,
                              shuffle=True)
    
    val_loader   = DataLoader(val_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=6, 
                              persistent_workers=True, 
                              shuffle=False)
    
    for _ in range(args.load_epoch):
        for __ in range(len(train_loader)):
            scheduler.step()
    #///////////////////////#


    # 모델 저장 위치 
    model_path = './save/'
    model_num = 1
    model_file = model_path+'best_'+args.model+'_model' + str(model_num) + end_pth
    while (os.path.isfile(model_file)):
        model_num += 1
        model_file = model_path+'best_'+args.model+'_model' + str(model_num) + end_pth
    print(model_file)
    mo = open(model_file,"w") 
    # 파일을 닫습니다. 파일을 닫지 않으면 데이터 손실이 발생할 수 있습니다.
    mo.write('temp')
    mo.close() 

    # 모델 Loss 위치 
    loss_path = './loss/'
    loss_num = 1
    loss_file = loss_path+args.model+'_model' + str(loss_num) + end_pth
    while (os.path.isfile(loss_file)):
        loss_num += 1
        loss_file = loss_path+args.model+'_model' + str(loss_num) + end_pth
    print(loss_file)
    lo = open(loss_file,"w") 
    # 파일을 닫습니다. 파일을 닫지 않으면 데이터 손실이 발생할 수 있습니다.
    lo.write('temp')
    lo.close() 



    print(f"running: {device}, \nModel: {args.model} \nepoch: {args.epoch},  \
            \nbatch: {args.batch_size}, \nlr: {args.lr} \nSummary: {args.summary} \
            \nloss: {args.loss}, \nNoise_train: {args.noise_train} \nNoise {args.noise} \
            \patch_size: {args.train_img_size}")

    # ?///////////////////////#


    # 모델 학습
    print("모델 학습 시작")
    train(args.epoch,noise = result_noise, save_val=bools[args.val_best_save], load_epoch=args.load_epoch)


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