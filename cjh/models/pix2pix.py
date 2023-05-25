import random
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as tf
from torchvision.transforms import ToTensor, Normalize, Compose
from os.path import join
from os import listdir
from torchsummary import summary
import time
import argparse
import models.DnCNN as DnCNN, models.ResNet as ResNet, models.RFDN as RFDN
import models.DRLN as DRLN, models.pix2pix as pix2pix, models.pix2pix2 as pix2pix2
import utils.randaugment as randaugment
from utils.param import param_check, seed_everything
import utils.vgg_loss, utils.vgg_perceptual_loss
from PIL import Image
import matplotlib.pyplot as plt
import gc
from math import log10
gc.collect()
torch.cuda.empty_cache()

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
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# 커스텀 데이터셋 클래스 정의
class CustomDataset(data.Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, patch_size = 128, transform=None, noisy_train = False):
        # super(Dataset, self).__init__() # 초기화 상속
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]# a는 건물 사진
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]# b는 Segmentation Mask
        self.transform = transform
        self.patch_size = patch_size
        self.noisy_train = noisy_train
        # self.image_filenames = [x for x in os.listdir(self.a_path)] # a 폴더에 있는 파일 목록

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
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
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

class loss_save():
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []

    def add(self, model, epoch, train_loss, val_loss, save_dir):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        torch.save(
            {
                "model": model,
                "epoch": self.epochs,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss
            }, save_dir
    )

def tensor_to_yuv(images):
    i =[]
    for image in images:
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
def train(num_epochs, noise = True, save_val=True, model_type=False):
    best_loss = 9999.0
    best_val_loss = 9999.0
    tem = 1
    total_iter = len(train_loader)
    loss_pth = loss_save()
    if not model_type:
        print(0)
        for epoch in range(args.epoch):
            model.train()
            epoch_time = time.time()
            running_loss = 0.0
            tot_mae = 0.0
            for iter, (noisy_images, clean_images) in enumerate(train_loader):
                noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
                optimizer.zero_grad()
                outputs = model(noisy_images)
                if tem:
                    tem=0
                    print(outputs.size(), clean_images.size())
                if noise:
                    loss = criterion(outputs, noisy_images-clean_images)
                else:
                    loss = criterion(outputs, clean_images)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * noisy_images.size(0)
                tot_mae += cal_mae_loss(tensor_to_yuv(outputs),tensor_to_yuv(clean_images)) / len(train_dataset)
                if (iter+1) % int(total_iter/4) == 0:
                    print(f"\t[{iter+1}/{total_iter}] \tlr: {optimizer.param_groups[0]['lr']} \tTrain_Loss: {loss.item():.4f}")
            scheduler.step()
            epoch_loss = running_loss / len(train_dataset)
            val_loss = val(noise)
            loss_pth.add(args.model,epoch,epoch_loss,val_loss,loss_file)
            print(f'Epoch {epoch+1}/{num_epochs} \tTime: {time.time()-epoch_time:.0f}초 \
                \tTrain_Loss: {epoch_loss:.4f} \tVal_Loss: {val_loss:.4f} \tMAE : {tot_mae:.5f}')

        # 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
            if save_val:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_file)
                    print(f"{epoch+1}epoch 모델 저장 완료")
            else:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), model_file)
                    print(f"{epoch+1}epoch 모델 저장 완료")
    else:
        for epoch in range(args.epoch):
            print('======================================================================================================')
            G.train()
            D.train()
            epoch_time = time.time()
            running_d_loss = 0.0
            running_g_loss = 0.0
            # tot_mae = 0.0
            for iter, (noisy_images, clean_images) in enumerate(train_loader):
                real_n, real_c = noisy_images.to(device), clean_images.to(device)
                real_label = torch.ones(1).cuda()
                fake_label = torch.zeros(1).cuda()

                fake_c = G(real_n) # G가 생성한 fake noisy mask

                #============= Train the discriminator =============#
                # train with fake
                fake_nc = torch.cat((real_n, fake_c), 1)
                pred_fake = D.forward(fake_nc.detach())
                loss_d_fake = criterionMSE(pred_fake, fake_label)

                # train with real
                real_nc = torch.cat((real_n, real_c), 1)
                pred_real = D.forward(real_nc)
                loss_d_real = criterionMSE(pred_real, real_label)
                
                # Combined D loss
                loss_d = (loss_d_fake + loss_d_real) * 0.5
                
                # Backprop + Optimize
                D.zero_grad()
                loss_d.backward()
                d_optimizer.step()

                #=============== Train the generator ===============#
                # First, G(A) should fake the discriminator
                fake_nc = torch.cat((real_n, fake_c), 1)
                pred_fake = D.forward(fake_nc)
                loss_g_gan = criterionMSE(pred_fake, real_label)

                # Second, G(A) = B
                loss_g_l1 = criterionL1(fake_c, real_c) * 10
                
                loss_g = loss_g_gan + loss_g_l1
                
                # Backprop + Optimize
                G.zero_grad()
                D.zero_grad()
                loss_g.backward()
                g_optimizer.step()
                running_d_loss += loss_d.item() * noisy_images.size(0)
                running_g_loss += loss_g.item() * noisy_images.size(0)
                # tot_mae += cal_mae_loss(tensor_to_yuv(noisy_images),tensor_to_yuv(clean_images))
                if (iter+1) % int(total_iter/2) == 0:
                    print('Epoch [%d/%d], Step[%d/%d], %.0f초, d_loss: %.4f, g_loss: %.4f'
                        % (epoch, args.epoch, iter, len(train_loader),time.time()-epoch_time, loss_d.item(), loss_g.item()))
                    # show_images(denorm(real_n.squeeze()), denorm(real_c.squeeze()), denorm(fake_c.squeeze()))
            epoch_d_loss = running_d_loss / len(train_dataset)
            epoch_g_loss = running_g_loss / len(train_dataset)
            print(f'\tepoch_d_loss {epoch_d_loss:.4f}, \tepoch_g_loss {epoch_g_loss:.4f}')
            loss_pth.add(args.model,epoch,epoch_d_loss,epoch_g_loss,loss_file)
            if epoch_g_loss < best_loss:
                    best_loss = epoch_g_loss
                    torch.save(G.state_dict(), model_file)
                    torch.save(D.state_dict(), model_file.replace('_g','_d'))
                    print(f"\t{epoch+1}epoch 모델 저장 완료")
            print('======================================================================================================')


def val(noise = True):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for noisy_images, clean_images in val_loader:
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            outputs = model(noisy_images)
            if noise:
                loss = criterion(outputs, noisy_images-clean_images)
            else:
                loss = criterion(outputs, clean_images)
            running_loss += loss.item() * noisy_images.size(0)
        epoch_loss = running_loss / len(val_dataset)
    return epoch_loss

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

    m = args.model
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
        args.lr = 5e-4
    elif m == 'DRLN':
        model = DRLN.DRLN().to(device)
        args.lr = 1e-4
    elif m == 'pix2pix':
        G = pix2pix.Generator().to(device)
        D = pix2pix.Discriminator().to(device)
        # args.batch_size = 1
        end_pth = '_g.pth'
    elif m == 'pix2pix2':
        G = pix2pix2.GeneratorUNet().to(device)
        D = pix2pix2.Discriminator().to(device)
        # args.batch_size = 1
        end_pth = '_g.pth'
    else:
        model = DnCNN.DnCNN().to(device)
    if m == 'pix2pix' or m == 'pix2pix2': 
        print('성공)')
        model_type = True
        print('총 : ',param_check(G) + param_check(D))
        print('총 : ',param_check(G, True) + param_check(D, True))
        if args.summary == 'True' or args.summary == 'true':
            print(summary(G, (3, 128, 128)))
            print(summary(D, (6, 128, 128)))
        criterionL1 = nn.L1Loss().to(device)
        criterionMSE = nn.MSELoss().to(device)

        # Setup optimizer
        g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    else:
        model_type = False
        param_check(model)
        param_check(model, True)
        if args.summary == 'True' or args.summary == 'true':
            print(summary(model, (3, 128, 128)))

         # 손실 함수와 최적화 알고리즘 설정
        if args.loss == 0:
            criterion = nn.MSELoss()
        elif args.loss == 1:
            criterion = utils.vgg_loss.WeightedLoss([utils.vgg_loss.VGGLoss(model='vgg16',shift=2),
                                                    nn.MSELoss(),
                                                    utils.vgg_loss.TVLoss(p=1)],
                                                    [1, 40, 10]).to(device)
        elif args.loss == 2:
            vgg_model = 'vgg16'
            print('! vgg model :' , vgg_model)
            criterion = utils.vgg_perceptual_loss.VGGPerceptualLoss(model=vgg_model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2e5, gamma=0.5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 'min')
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

    val_transform = Compose([
        # BilateralBlur(args.train_img_size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 커스텀 데이터셋 인스턴스 생성
    dataset = CustomDataset(noisy_image_paths, clean_image_paths, args.train_img_size, transform=train_transform, noisy_train = bools[args.noise_train])
    
    # val 분할
    train_size = int(len(dataset)*(1-args.val))
    val_size = int(len(dataset)*(args.val))
    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])
    print(f"Train Size : {len(train_dataset)} \tValidattion Size : {len(val_dataset)}")
    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=4, 
                              persistent_workers=True, 
                              shuffle=True)
    
    val_loader   = DataLoader(val_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=4, 
                              persistent_workers=True, 
                              shuffle=False)
    

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
            \nloss: {loss_list[args.loss]}, \nNoise_train: {args.noise_train}")

    # ?///////////////////////#


    # 모델 학습
    print("모델 학습 시작")
    train(args.epoch,noise = result_noise, save_val=bools[args.val_best_save], model_type=model_type)


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