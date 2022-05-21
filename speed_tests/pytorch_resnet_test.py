import torch
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import time


device = 'mps'

if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    model.eval()

    input_shape = [-1, 3, 224, 224]
    x = np.random.rand(32, 3, 224, 224).astype('float32')
    
    num_steps = 50
    start = time.time()
    model.to(device)

    for i in tqdm(range(num_steps)):
        x_ = torch.from_numpy(x)
        x_ = x_.to(device)
        result = model(x_).cpu().detach().numpy()
        pass
    
    print(f'{num_steps} cost {time.time()-start} seconds')
    pass
