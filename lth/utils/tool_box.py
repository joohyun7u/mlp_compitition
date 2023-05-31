import random, numpy as np, cv2, time, csv
import os
from os import listdir
from os.path import join

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import torchsummary

device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

############## 학습 관련 #################

class TrainDataset(data.Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, patch_size, noisy_transform=None, clean_transform=None):
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]
        self.noisy_transform = noisy_transform
        self.clean_transform = clean_transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
        clean_image = load_img(self.clean_image_paths[index])

        H, W, _ = noisy_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        clean_image = clean_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

        # transform 적용
        if self.noisy_transform:
            noisy_image = self.noisy_transform(noisy_image)
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


class Trainer():
    def __init__(self, model, version, model_name, total_iteraion, 
                 train_data_loader, valid_data_loader , validation_checkpoint, 
                 validation_output_dir, optimizer, criterion, scheduler, 
                 model_save_dir ,load_pth_name = None, current_step = 0):
        
        self.model = model
        self.version = version
        self.model_name = model_name
        self.total_iteraion = total_iteraion
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.validation_output_dir = validation_output_dir
        self.validation_checkpoint = validation_checkpoint
        self.optimizer = optimizer 
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_save_dir = model_save_dir
        self.load_pth_name = load_pth_name
        self.current_step = current_step
        
        if self.load_pth_name:
            self.model.load_state_dict(torch.load(join(model_save_dir, load_pth_name + ".pth")))

        self.best_val_loss = 9999.0

        for _ in range(self.current_step):
            self.scheduler.step()

    def train(self):
        for epoch in range(500000):
            
            epoch_time = time.time()

            running_loss = 0.0
            for iter, (noisy_images, clean_images) in enumerate(self.train_data_loader):
                
                self.current_step += 1
                
                self.model.train()
                noisy_images, clean_images =  noisy_images.to(device), clean_images.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(noisy_images)
                loss = self.criterion(outputs,clean_images)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item() * noisy_images.size(0)

                if self.current_step % self.validation_checkpoint == 0:
                    self.validation()   
    
            train_loss = running_loss / len(self.train_data_loader)

            print(f'step {self.current_step}/{self.total_iteraion} Time: {time.time()-epoch_time:.0f}s Train_Loss: {train_loss:.4f} lr: {self.scheduler.get_lr()}')


    def validation(self):
        self.model.eval()
        running_loss = 0.0
        
        worst_loss = 0.0
        worst_loss_iter = 0
        
        for iter, (noisy_images, clean_images) in enumerate(self.valid_data_loader):
            noisy_image, clean_images = noisy_images.to(device), clean_images.to(device)
            outputs = self.model(noisy_image)
            loss = self.criterion(outputs, clean_images)
            running_loss += loss.item() * noisy_images.size(0)

            if (loss.item() * noisy_images.size(0) > worst_loss):
                worst_loss = loss.item() * noisy_images.size(0)
                worst_loss_iter = iter

        epoch_loss = running_loss / len(self.valid_data_loader)
        print(f'step {self.current_step}/{self.total_iteraion}  Validation_loss: {epoch_loss:.4f}')
        
        for iter, (noisy_images, clean_images) in enumerate(self.valid_data_loader):
            if (iter != worst_loss_iter):
                continue
            noisy_image, clean_images = noisy_images.to(device), clean_images.to(device)
            outputs = self.model(noisy_image)

            original_image = noisy_images[0]
            original_image = torch.clamp(original_image, 0, 1)
            original_image = transforms.ToPILImage()(original_image)

            denoised_image = outputs[0]
            denoised_image = torch.clamp(denoised_image, 0, 1)  
            denoised_image = transforms.ToPILImage()(denoised_image)

            clean_image = clean_images[0]
            clean_image = torch.clamp(clean_image, 0, 1)  
            clean_image = transforms.ToPILImage()(clean_image)

            original_filename = join(self.validation_output_dir, f"{self.current_step}_{worst_loss}_scan" + '.png')
            original_image.save(original_filename)

            denoised_filename = join(self.validation_output_dir, f"{self.current_step}_{worst_loss}_denoised" + '.png')
            denoised_image.save(denoised_filename)

            clean_filename = join(self.validation_output_dir, f"{self.current_step}_{worst_loss}_clean"+ '.png')
            clean_image.save(clean_filename)

            print(f'Saved scanned  image: {original_filename}')
            print(f'Saved denoised image: {denoised_filename}')  
            print(f'Saved clean image: {clean_filename}')  

            break
            
        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            torch.save(self.model.state_dict(), self.model_save_dir + self.model_name + self.version + ".pth")

    
    def train_info(self):

        criterion_name = str(self.criterion).split('\n')[0]
        optimizer_name = str(self.optimizer).split('\n')[0]

        print(f"========================= TRAIN INFO =========================== \
            \nRunning on: {device}\
            \nModel: {self.model_name}\
            \nVersion: {self.version}\
            \nTotal_iteration: {self.total_iteraion}\
            \nLr_init: {self.scheduler.get_lr()}\
            \nCriterion: {criterion_name}\
            \nOptimizer: {optimizer_name}\
            \n================================================================")
        
        torchsummary.summary(self.model,input_size=(3,32,32))
        param_check(model=self.model)        


##################### 테스트 관련 ######################################

class TestDatastLoader(data.Dataset):
    def __init__(self, noisy_image_paths, noisy_transform):
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.noisy_transform = noisy_transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])
        
        noisy_image = self.noisy_transform(noisy_image)

        return noisy_image, noisy_image_path

class Tester():
    def __init__(self, model, model_name, image_size, window_size, model_save_dir ,model_pth_name, test_data_loader, image_output_dir, display_num):
        self.model = model
        self.model_name = model_name
        self.image_size = image_size
        self.window_size = window_size
        self.model_save_dir = model_save_dir
        self.model_pth_name = model_pth_name
        self.test_data_loader = test_data_loader
        self.image_output_dir = image_output_dir
        self.display_num = display_num

        # 모델 파라미터 로딩
        pth_filename = join(model_save_dir, model_pth_name + '.pth')
        print(f"pth_filename: {pth_filename}")
        model.load_state_dict(torch.load(pth_filename))
        model.eval()

        # 비교할 스캔 이미지, 디노이즈드 이미지 경로
        self.scan_output_dir = join(self.image_output_dir, 'scan/')
        self.denoise_output_dir = join(self.image_output_dir, 'denoise/')

        if not os.path.exists(self.scan_output_dir):
            os.makedirs(self.scan_output_dir)

        if not os.path.exists(self.denoise_output_dir):
            os.makedirs(self.denoise_output_dir)


    def test(self):
        self.model.eval()
        for iter, (noisy_image, noisy_image_path) in enumerate(self.test_data_loader):
            
            noisy_image = noisy_image.to(device)

            with torch.no_grad():
                # pad input image to be a multiple of window_size
                img_lq = noisy_image
                h_old, w_old = self.image_size
                h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
                w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = self.model(img_lq)
                denoised_image = output[..., :h_old, :w_old]
                denoised_image = torch.clamp(denoised_image,0,1)
            
            output_filename = noisy_image_path[0]

            denoised_image = transforms.ToPILImage()(denoised_image.squeeze(0))
            denoised_filename = join(self.denoise_output_dir, output_filename.replace('\\','/').split('/')[-1][:-4] + '.png')
            denoised_image.save(denoised_filename)


            if iter > self.display_num:
                continue

            noisy_image = torch.clamp(noisy_image,0,1)
            noisy_image = transforms.ToPILImage()(noisy_image.squeeze(0))
            scaned_filename = join(self.scan_output_dir, output_filename.replace('\\','/').split('/')[-1][:-4] + '.png')
            noisy_image.save(scaned_filename)
        
            print(f'Saved scanned  image: {scaned_filename}')
            print(f'Saved denoised image: {denoised_filename}')

    def make_csv(self):
        # CSV 파일 생성
        output_file = join(self.image_output_dir ,self.model_pth_name+'.csv')


        self.denoise_output_dir = join(self.image_output_dir, 'denoise/')
        # 폴더 내 이미지 파일 이름 목록을 가져오기
        file_names = os.listdir(self.denoise_output_dir)
        file_names.sort()

        
        # CSV 파일을 작성하기 위해 오픈
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image File', 'Y Channel Value'])

            for file_name in file_names:
                # 이미지 로드
                image_path = os.path.join(self.denoise_output_dir, file_name)
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



################ 후처리 필터 ######################

class ProccessDatastLoader(data.Dataset):
    def __init__(self, denoised_image_paths, denoised_transform):
        self.denoised_image_paths = [join(denoised_image_paths, x) for x in listdir(denoised_image_paths)]
        self.denoised_transform = denoised_transform

    def __len__(self):
        return len(self.denoised_image_paths)

    def __getitem__(self, index):
        
        denoised_image_path = self.denoised_image_paths[index]
        denoised_image = load_img(self.denoised_image_paths[index])
        
        denoised_image = self.denoised_transform(denoised_image)

        return denoised_image, denoised_image_path

class Bilateral():
    def __init__(self, denoised_dataloder, image_output_dir, model_pth_name):

        self.denoised_dataloder = denoised_dataloder
        self.image_output_dir = image_output_dir
        self.model_pth_name = model_pth_name

        self.bilateral_output_dir = join(self.image_output_dir, 'bilateral/')

        if not os.path.exists(self.bilateral_output_dir):
            os.makedirs(self.bilateral_output_dir)


    def make_bilateral_img(self, kernel_size, sigma_c, sigma_s):
        # 바이래터럴 필터 적용

        for iter, (denoised_image, denoised_image_path) in enumerate(self.denoised_dataloder):

            output_filename = denoised_image_path[0].replace('\\','/').split('/')[-1][:-4] + '.png'

            denoised_image = denoised_image.squeeze(0)
            denoised_image = torch.clamp(denoised_image,0,1)
            denoised_image = transforms.ToPILImage()(denoised_image)
            denoised_image = np.array(denoised_image)

            output = cv2.bilateralFilter(denoised_image, kernel_size, sigma_c, sigma_s)
            
            output = np.clip(output,0,255)
            output = transforms.ToPILImage()(output)

            output_filename = join(self.bilateral_output_dir, output_filename)
            output.save(output_filename)

            print(f'Saved bilateral denoised image: {output_filename}')

    def make_csv(self):
        file_names = os.listdir(self.bilateral_output_dir)
        file_names.sort()

        csv_file = join(self.image_output_dir ,self.model_pth_name+'_bilateral.csv')
        
        # CSV 파일을 작성하기 위해 오픈
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image File', 'Y Channel Value'])

            for file_name in file_names:
                # 이미지 로드
                image_path = os.path.join(self.bilateral_output_dir, file_name)
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




############## 유틸리티 ################
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def param_check(model):
    isModelSatisfiesCondition = None

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    if int(pytorch_total_params) > 10000000:
        print(f'@@@@@@@@@@@@@ IT IS OVER THEN 10 MILLIONS {pytorch_total_params-10000000} PARAMETERS EXCEEDS@@@@@@@@@@@@@')
        isModelSatisfiesCondition = False
    else:
        print(f'@@@@@@@@@@@@@ IT IS LOWER THAN 10 MILLIONS {10000000-pytorch_total_params} PARAMETERS LEFT@@@@@@@@@@@@@')
        isModelSatisfiesCondition = True

    return isModelSatisfiesCondition

