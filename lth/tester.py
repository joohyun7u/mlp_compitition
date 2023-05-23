import argparse, importlib
from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

from utils import tool_box as T
from utils.custom_transforms import RGBtoYcrcb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--datasets_dir',   type=str)
    parser.add_argument('--model',          type=str)
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--output_dir',     type=str)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

    print(f"==================== TEST  INFO =================== \
        \nrunning: {device}\
        \nModel: {args.model}\
        \noutput_dir: {args.output_dir}\
        \n===================================================")

    T.seed_everything(42)

    # 하이퍼파라미터 설정
    dataset_dir = args.datasets_dir
    model_name = args.model
    model_save_path = args.model_save_dir
    output_dir = args.output_dir

    # 모델 로딩
    model = importlib.import_module('.' + model_name, '.models').model
    model.load_state_dict(torch.load(join(model_save_path,model_name + ".pth")))
    model.eval()

    # 전처리
    noisy_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 데이터셋 설정
    noisy_dataset = T.TestDatastLoader(
        noisy_image_paths = args.datasets_dir, 
        noisy_transform = noisy_transform
    )

    # 데이터 로더 설정
    noisy_loader = DataLoader(noisy_dataset, batch_size=1, shuffle=False)

    # 테스트
    tester = T.Tester(
        model = model,
        model_name = model_name,
        test_data_loader = noisy_loader,
        output_dir = output_dir
    )
    tester.test()