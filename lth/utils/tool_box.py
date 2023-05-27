import random, numpy as np, cv2, time
import os
from os import listdir
from os.path import join

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

##############트레이너 관련################

class CustomDataset(data.Dataset):
    def __init__(self, noisy_image_paths, noise_image_paths, patch_size = 128, noisy_transform=None, noise_transform=None):
        "note that It is not noisy_image, It is noise_image"
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.noise_image_paths = [join(noise_image_paths, x) for x in listdir(noise_image_paths)]
        self.noisy_transform = noisy_transform
        self.noise_transform = noise_transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.noise_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
        noise_image = load_img(self.noise_image_paths[index])

        H, W, _ = noisy_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        noise_image = noise_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

        # transform 적용
        if self.noisy_transform:
            noisy_image = self.noisy_transform(noisy_image)
        if self.noise_transform:
            noise_image = self.noise_transform(noise_image)


        # 이미지 랜덤 수평 플립
        if torch.rand(1) < 0.5:
            noisy_image = F.hflip(noisy_image)
            noise_image = F.hflip(noise_image)
        
        # 이미지 랜덤 수직 플립
        if torch.rand(1) < 0.5:
            noisy_image = F.vflip(noisy_image)
            noise_image = F.vflip(noise_image)
        
        return noisy_image, noise_image
    
    
class Trainer():
    def __init__(self, model, version, model_name, num_epochs, train_data_loader, valid_data_loader , optimizer, criterion, scheduler, model_save_dir,loss_save_dir):
        self.model = model
        self.version = version
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.optimizer = optimizer 
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_save_dir = model_save_dir
        self.loss_save_dir = loss_save_dir

    def train(self):
        best_val_loss = 9999.0
        loss_pth = loss_manager()

        timer = Timer()
        timer.start()

        for epoch in range(self.num_epochs):
            self.model.train()
            
            epoch_time = time.time()

            running_loss = 0.0
            for iter, (noisy_images, noise_images) in enumerate(self.train_data_loader):
                noisy_images, noise_images =  noisy_images.to(device), noise_images.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(noisy_images)
                loss = self.criterion(outputs,noise_images)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * noisy_images.size(0)
    
            train_loss = running_loss / len(self.train_data_loader)
            val_loss = self.val()
            self.scheduler.step()

            loss_pth.add(epoch,train_loss,val_loss)

            print(f'Epoch {epoch+1}/{self.num_epochs} Time: {time.time()-epoch_time:.0f}s Train_Loss: {train_loss:.4f} Val_Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_dir + self.model_name + self.version + ".pth")


        loss_pth.save(self.model_name + self.version, self.loss_save_dir)
        timer.stop()


    def val(self):
        self.model.eval()
        running_loss = 0.0
        for noisy_images, noise_images in self.valid_data_loader :
            noisy_image, noise_image = noisy_images.to(device), noise_images.to(device)
            outputs = self.model(noisy_image)
            loss = self.criterion(outputs, noise_image)
            running_loss += loss.item() * noisy_images.size(0)
        epoch_loss = running_loss / len(self.valid_data_loader)
        return epoch_loss



##############테스트 관련################

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
    def __init__(self, model, model_name, model_save_dir ,model_pth_name, test_data_loader, image_output_dir, display_num):
        self.model = model
        self.model_name = model_name
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

            if iter+1 > self.display_num:
                break

            noisy_image = noisy_image.to(device)
            noise = self.model(noisy_image)

            denoised_image = noisy_image - noise
        
            denoised_image = denoised_image.cpu().squeeze(0)
            denoised_image = torch.clamp(denoised_image, 0, 1)  
            denoised_image = transforms.ToPILImage()(denoised_image)

            scanned_image = noisy_image.cpu().squeeze(0)
            scanned_image = torch.clamp(scanned_image, 0, 1)
            scanned_image = transforms.ToPILImage()(scanned_image)

            output_filename = noisy_image_path[0]

            denoised_filename = join(self.denoise_output_dir, output_filename.replace('\\','/').split('/')[-1][:-4] + '.png')
            denoised_image.save(denoised_filename)

            scaned_filename = join(self.scan_output_dir, output_filename.replace('\\','/').split('/')[-1][:-4] + '.png')
            scanned_image.save(scaned_filename)
        
            print(f'Saved scanned  image: {scaned_filename}')
            print(f'Saved denoised image: {denoised_filename}')


############## 유틸리티 ################
class Timer():    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        return
    
    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()   
        
        training_time = self.end_time - self.start_time

        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        hours = int(minutes // 60)
        minutes = int(minutes % 60)

        print(f"total training duration: {hours}H {minutes}M {seconds}S")

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

class loss_manager():
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []

    def add(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        
    def save(self, model_name, save_dir):
        torch.save(
            {
                "model": model_name,
                "epoch": self.epochs,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss
            }, save_dir + str(model_name) + '.pth'
        )

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