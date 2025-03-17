#%%
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils    
import layers
from models import Sequential
from keras.utils import to_categorical


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]

    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28,28,1).astype("float32") / 255
y_train = to_categorical(y_train, num_classes=10)
y_train = y_train.reshape(60000, 10, 1)

network = [
    layers.BatchNormalization(),
    layers.Convolution(forward_verbose=False, backward_verbose=False, kernel_size=5, n_kernels=24, stride=(1,1), activation="relu"),

    layers.Convolution(forward_verbose=False, backward_verbose=False, kernel_size=5, n_kernels=48, stride=(1,1), activation="relu"),

    layers.Convolution(forward_verbose=False, backward_verbose=False, kernel_size=5, n_kernels=64, stride=(1,1), activation="relu"),

    layers.Flatten(forward_verbose=False, backward_verbose=False),

    layers.Dense(forward_verbose=False, backward_verbose=False, units=256, activation="relu"),

    layers.Dense(forward_verbose=False, backward_verbose=False, units=10, activation="softmax")
]


model = Sequential(network, learning_rate=0.01)
model.compile(loss='categorical_crossentropy')
model.fit(x_train[0:60000], y_train[0:60000], epochs=1, batch_size=32)

#%%
x = np.array([[0], [1], [2], [3]], dtype=np.float32)
y = np.array([[0], [2], [4], [6]], dtype=np.float32)

network = [
    layers.Dense(forward_verbose=False, backward_verbose=False, units=1, activation="softmax")
]

#%%
import dill

with open("cnn_model.pkl", "wb") as arquivo:
    dill.dump(model, arquivo)

#%%
import dill
with open("cnn_model.pkl", "rb") as arquivo:
    model = dill.load(arquivo)
