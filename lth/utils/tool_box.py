import random, numpy as np, cv2, time

import os
from os import listdir
from os.path import join

import torch
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

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
    
class Trainer():
    def __init__(self, model, model_name, num_epochs, train_data_loader, valid_data_loader , optimizer, criterion,model_save_dir,loss_save_dir):
        self.model = model
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.optimizer = optimizer
        self.criterion = criterion
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
            
            for iter, (noisy_images, clean_images) in enumerate(self.train_data_loader):
                noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(noisy_images)
                loss = self.criterion(outputs, noisy_images-clean_images)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * noisy_images.size(0)
    
            train_loss = running_loss / len(self.train_data_loader)
            val_loss = self.val()

            loss_pth.add(epoch,train_loss,val_loss)

            print(f'Epoch {epoch+1}/{self.num_epochs} Time: {time.time()-epoch_time:.0f}s Train_Loss: {train_loss:.4f} Val_Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_dir + self.model_name + ".pth")


        loss_pth.save(self.model_name, self.loss_save_dir)
        timer.stop()


    def val(self):
        self.model.eval()
        running_loss = 0.0
        for noisy_images, clean_images in self.valid_data_loader :
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            outputs = self.model(noisy_images)
            loss = self.criterion(outputs, noisy_images-clean_images)
            running_loss += loss.item() * noisy_images.size(0)
        epoch_loss = running_loss / len(self.valid_data_loader)
        return epoch_loss


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

# 전체 코드 랜덤 시드 설정
def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# 이미지 로딩 함수
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
