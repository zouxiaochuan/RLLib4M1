import torch
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import time
import platform


def run(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    input_shape = [-1, 3, 224, 224]
    x = np.random.rand(32, 3, 224, 224).astype('float32')
    x = torch.from_numpy(x).to(device)
    num_steps = 50
    start = time.time()
    model.to(device)

    for i in tqdm(range(num_steps)):
        result = model(x)
        pass
    
    print(f'{device} {num_steps} cost {time.time()-start} seconds')


if __name__ == '__main__':
    if platform.system().startswith('Darwin'):
        devices = ['cpu', 'mps']
    else:
        devices = ['cpu', 'cuda']
        pass
    for device in devices:
        run(device)
        pass
    pass
