import torch
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import time


if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    model.eval()

    input_shape = [-1, 3, 224, 224]
    x = np.random.rand(32, 3, 224, 224).astype('float32')
    
    num_steps = 50
    start = time.time()
    for i in tqdm(range(num_steps)):
        result = model(torch.from_numpy(x)).detach().numpy()
        pass
    
    print(f'{num_steps} cost {time.time()-start} seconds')
    pass
