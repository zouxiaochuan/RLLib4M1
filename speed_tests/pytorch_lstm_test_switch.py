import torch
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import time
import platform

INPUT_SIZE = 1024
HIDDEN_SIZE = 512

class LSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.layer = torch.nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=4)
        # self.output0 = torch.nn.Sigmoid()#(hidden_size, 1)

    def forward(self, x):
        x, h = self.layer(x)
        return x

    pass


def run(device):
    model = LSTM(INPUT_SIZE, HIDDEN_SIZE)
    # model = models.resnet50(pretrained=True)
    model.eval()
    x = np.random.rand(128, 1024).astype('float32')
    # x = torch.from_numpy(x).to(device)

    num_steps = 1000
    start = time.time()
    model.to(device)
    for i in tqdm(range(num_steps)):
        x_ = torch.from_numpy(x).to(device)
        result = model(x_).cpu()
        pass

    print(f'{device} {num_steps} cost {time.time()-start} seconds')
    pass

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
