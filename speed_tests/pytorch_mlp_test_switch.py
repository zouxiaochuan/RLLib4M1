import torch
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import time


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.fc0 = torch.nn.Linear(in_dim, hidden_size)
        self.tanh0 = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.tanh1 = torch.nn.Tanh()
        # self.output0 = torch.nn.Sigmoid()#(hidden_size, 1)
        self.output0 = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.fc0(x)
        x = self.tanh0(x)
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.output0(x)
        return x

    pass


def run(device):
    model = MLP(1024, 512)
    # model = models.resnet50(pretrained=True)
    model.eval()
    x = np.random.rand(32, 1024).astype('float32')
    # x = torch.from_numpy(x).to(device)

    num_steps = 10000
    start = time.time()
    model.to(device)
    for i in tqdm(range(num_steps)):
        x_ = torch.from_numpy(x).to(device)
        result = model(x_).cpu()
        pass

    print(f'{device} {num_steps} cost {time.time()-start} seconds')
    pass

if __name__ == '__main__':
    if os.system().startswith('Darwin'):
        devices = ['cpu', 'mps']
    else:
        devices = ['cpu', 'cuda']
        pass
    for device in devices:
        run(device)
        pass
    pass