import torch, random, os
import numpy as np

def param_check(model, grad=False):
    if grad:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('grad인 parameter 개수기준임')
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('모든 parameter 개수기준임')
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 10000000:
        print(f'!!!!!!10M 1천만보다 {pytorch_total_params-10000000}개 초과했음!!!!!!')
    else:
        print(f'10M 1천만보다 {10000000-pytorch_total_params}개 여유 있음...')
    return pytorch_total_params

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True