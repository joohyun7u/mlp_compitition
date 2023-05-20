import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import os
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

num_epochs = 100
learning_rate =0.1
batch_size = 110

train_data_dir = '/local_datasets/ImageNet/train'
test_data_dir = '/local_datasets/ImageNet/val'

def cal_norm(dataset):
    mean_ = np.array([np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset])
    print("mean np end")
    mean_r = mean_[:,0].mean()
    print('mean r end')
    mean_b = mean_[:,1].mean()
    print('mean g end')
    mean_g = mean_[:,2].mean()
    print('mean b end')

    print('mean end')
    std_ = np.array([np.std(x.numpy(), axis=(1,2)) for x,_ in dataset])
    print("std np end")
    std_r = std_[:,0].mean()
    print("std r end")
    std_b = std_[:,1].mean()
    print("std g end")
    std_g = std_[:,2].mean()
    print("std b end")

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)


# norm_dataset = torchvision.datasets.ImageFolder(train_data_dir,
#                                                 transform=transforms.ToTensor()
#                                                 )

# print("calstart")
# mean_, std_ = cal_norm(norm_dataset)
# print(mean_, std_)
# 직접 계산한 mead, std 값
mean_, std_ = [0.48108798, 0.40784082, 0.45745823],  [0.23348042, 0.2302363, 0.22942752]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean_, std_)]
)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean_, std_)]
)

classes = []
img_per_class = []
# for folder in os.listdir(data_dir+'consolidated'):/
for folder in os.listdir(train_data_dir):    
    classes.append(folder)
    img_per_class.append(len(os.listdir(f'{train_data_dir}/{folder}')))
num_classes = len(classes)
df = pd.DataFrame({'Classes':classes, 'Examples':img_per_class})


# splitting the data into train/validation/test sets
# data = torchvision.datasets.ImageFolder(data_dir)
# train_size = int(len(data)*0.9)
# val_size = int((len(data)-train_size))
# train_dataset,test_dataset = torch.utils.data.random_split(data,[train_size,val_size])

train_dataset = torchvision.datasets.ImageFolder(train_data_dir,
                                                 transform=train_transform
                                                 )
test_dataset = torchvision.datasets.ImageFolder(test_data_dir,
                                                 transform=test_transform
                                                )
torch.manual_seed(3334)
print(f'train size: {len(train_dataset)}\nval size: {len(test_dataset)}')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=4,
                                           persistent_workers=True,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          persistent_workers=True,
                                          shuffle=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=1000, init_weights=True):
        super().__init__()

        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 1)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        # x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.conv6(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3,4,23,3])

def resnet152():
    return ResNet(BottleNeck, [3,8,36,3])


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

total_step = len(train_loader)
def train(epoch):
    model.train()
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        #print(images.size())
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        # correct += (predicted == labels).cpu().sum().item()

        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], step [{i+1}/{total_step}] Loss: {loss.item():.4f}')
    print(f'\tTrain Accuracy: {correct}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.4f})%')

    if (epoch+1) % 15 == 0:
        global curr_lr 
        curr_lr /= 10
        update_lr(optimizer, curr_lr)


def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # correct += (predicted == labels).cpu().sum().item()
        print(f'Accuracy of the model on the test images: {100*correct/total} %')

if __name__ == '__main__':
    model = resnet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        train(epoch)
        test()