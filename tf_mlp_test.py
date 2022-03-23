

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras as keras
import numpy as np
from tqdm import tqdm


class MLP(Model):
  def __init__(self, input_dim, hidden_size):
    super(MLP, self).__init__()
    # self.input0 = keras.layers.Input(shape=(input_dim,))
    self.fc0 = keras.layers.Dense(hidden_size)
    self.act0 = keras.layers.Activation(tf.nn.tanh)
    self.fc1 = keras.layers.Dense(hidden_size)
    self.act1 = keras.layers.Activation(tf.nn.tanh)
    self.output1 = keras.layers.Dense(2)

  def call(self, x):
    x = self.fc0(x)
    x = self.act0(x)
    x = self.fc1(x)
    x = self.act1(x)
    return self.output1(x)

if __name__ == '__main__':
    
    model = MLP(1024, 512)

    x_in = np.random.rand(2, 1024)

    for i in tqdm(range(100)):
        x = model(x_in)
        pass
    pass
