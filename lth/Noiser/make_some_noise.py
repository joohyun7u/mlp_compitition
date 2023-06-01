import argparse, importlib

from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

from utils import tool_box as T

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--datasets_dir',   type=str)
    parser.add_argument('--model',          type=str)
    parser.add_argument('--model_version',  type=str)
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--model_pth_name', type=str)
    parser.add_argument('--output_dir',     type=str)
  
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

    # 하이퍼파라미터 설정
    dataset_dir = args.datasets_dir
    model_name = args.model
    model_save_path = args.model_save_dir
    model_pth_name = args.model_pth_name
    output_dir = args.output_dir

    image_output_dir = join(output_dir,model_pth_name)

    # 모델 로딩
    model = importlib.import_module('.' + model_name, '.models').model
    model.load_state_dict(torch.load(join(model_save_path, model_pth_name + ".pth")))
    model.eval()

    # 전처리
    clean_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 데이터셋 설정
    clean_set = T.CleanDatastLoader(
        clean_image_paths = dataset_dir,
        clean_transform = clean_transform 
    )

    # 데이터 로더 설정
    clean_loader = DataLoader(clean_set, batch_size=1, shuffle=False, num_workers=4)
    
    noiser = T.Noiser(
        model = model,
        model_name = model_name,
        image_size = (512,512),
        window_size = 32, 
        model_save_dir = model_save_path,
        model_pth_name = model_pth_name,
        clean_data_loader = clean_loader,
        image_output_dir = image_output_dir,
    )
    
    noiser.make_some_noise()