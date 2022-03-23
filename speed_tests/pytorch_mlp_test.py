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




if __name__ == '__main__':
    model = MLP(1024, 512)
    # model = models.resnet50(pretrained=True)
    model.eval()

    # input_shape = [1, 3, 224, 224]
    input_shape = [-1, 1024]
    # x = np.random.rand(1, 3, 224, 224).astype('float32')
    x = np.random.rand(32, 1024).astype('float32')
    

    num_steps = 10000
    start = time.time()
    for i in tqdm(range(num_steps)):
        result = model(torch.from_numpy(x)).detach().numpy()
        pass

    print(f'{num_steps} cost {time.time()-start} seconds')
    pass
