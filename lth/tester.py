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
    parser.add_argument('--display_num',    type=int)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

    print(f"==================== TEST  INFO =================== \
        \nrunning: {device}\
        \nModel: {args.model}\
        \nVersion: {args.model_version}\
        \noutput_dir: {args.output_dir}\
        \ndisplay_num: {args.display_num}\
        \n===================================================")

    T.seed_everything(42)

    # 하이퍼파라미터 설정
    dataset_dir = args.datasets_dir
    model_name = args.model
    model_save_path = args.model_save_dir
    model_pth_name = args.model_pth_name
    output_dir = args.output_dir
    display_num = args.display_num

    image_output_dir = join(output_dir,model_pth_name)


    # 모델 로딩
    model = importlib.import_module('.' + model_name, '.models').model
    model.load_state_dict(torch.load(join(model_save_path, model_pth_name + ".pth")))
    model.eval()

    # 전처리
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 데이터셋 설정
    test_dataset = T.TestDatastLoader(
        noisy_image_paths = dataset_dir, 
        noisy_transform = test_transform
    )

    # 데이터 로더 설정
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 테스트
    tester = T.Tester(
        model = model,
        model_name = model_name,
        image_size = (512,512),
        window_size = 64, 
        model_save_dir = model_save_path,
        model_pth_name = model_pth_name,
        test_data_loader = test_loader,
        image_output_dir = image_output_dir,
        display_num = display_num
    )
    
    tester.test()

    tester.make_csv()