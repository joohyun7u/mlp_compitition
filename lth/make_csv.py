import argparse, importlib, cv2, csv, os, numpy as np

from os.path import join
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Normalize, Compose
import torchsummary

from utils import tool_box as T

device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--datasets_dir',   type=str)
    parser.add_argument('--csv_dir',        type=str)
    parser.add_argument('--model_name',     type=str)
    parser.add_argument('--model_pth_dir',  type=str)
    parser.add_argument('--model_pth_name', type=str)
    args = parser.parse_args()

T.seed_everything(42)

# 하이퍼파라미터 설정
dataset_dir = args.datasets_dir
csv_dir = args.csv_dir
model_name = args.model_name
model_pth_dir = args.model_pth_dir
model_pth_name = args.model_pth_name
image_output_dir = join(csv_dir, model_pth_name)


# 모델 로딩
model = importlib.import_module('.' + model_name, '.models').model
torchsummary.summary(model,input_size=(3,512,512))
T.param_check(model=model)

# 파라미터 로딩
pth_filename = join(model_pth_dir,model_pth_name + '.pth')
print(pth_filename)
model.load_state_dict(torch.load(pth_filename))
model.eval()


# 데이터셋 경로
noisy_data_path = join(dataset_dir, 'test/scan/')

print(noisy_data_path)
print(noisy_data_path)
print(noisy_data_path)
print(noisy_data_path)
print(noisy_data_path)

if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)

noisy_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터셋 로드 및 전처리
noisy_dataset = T.TestDatastLoader(noisy_data_path, noisy_transform=noisy_transform)

# 데이터 로더 설정
noisy_loader = DataLoader(noisy_dataset, batch_size=1, shuffle=False)


# 디노이징 및 이미지 저장
for noisy_image, noisy_image_path in noisy_loader:
    noisy_image = noisy_image.to(device)
    noise = model(noisy_image)

    denoised_image = noisy_image - noise
        
    # denoised_image를 CPU로 이동하여 이미지 저장
    denoised_image = denoised_image.cpu().squeeze(0)
    denoised_image = torch.clamp(denoised_image, 0, 1)  # 이미지 값을 0과 1 사이로 클램핑
    denoised_image = transforms.ToPILImage()(denoised_image)

    # Save denoised image
    output_filename = noisy_image_path[0]
    denoised_filename = join(image_output_dir , output_filename.replace('\\', '/').split('/')[-1][:-4] + '.png')
    print_format = cv2.cvtColor(np.array(denoised_image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(denoised_filename, print_format)

    print(f'Saved denoised image: {denoised_filename}')

# CSV 파일 생성

output_file = join(csv_dir ,model_pth_name+'.csv')

# 폴더 내 이미지 파일 이름 목록을 가져오기
file_names = os.listdir(image_output_dir)
file_names.sort()

# CSV 파일을 작성하기 위해 오픈
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image File', 'Y Channel Value'])

    for file_name in file_names:
        # 이미지 로드
        image_path = os.path.join(image_output_dir, file_name)
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