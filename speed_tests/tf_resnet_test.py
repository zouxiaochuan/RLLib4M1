

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras as keras
import numpy as np
from tqdm import tqdm
import time


if __name__ == '__main__':
    
    model = keras.applications.resnet50.ResNet50()

    x_in = np.random.rand(32, 224, 224, 3)

    num_steps = 50
    start = time.time()
    for i in tqdm(range(num_steps)):
        x = model(x_in)
        pass
    print(f'{num_steps} cost {time.time()-start} seconds')
    pass
