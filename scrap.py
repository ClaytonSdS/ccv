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

#%%

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
x_train = x_train.reshape(60000, 1,28,28)
x_train = x_train.astype("float32") / 255
y_train = to_categorical(y_train, num_classes=10)
y_train = y_train.reshape(60000, 10, 1)


#y_train_one_hot = to_categorical(y_train, num_classes=10)
#y_test_one_hot = to_categorical(y_test, num_classes=10)

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()
#x_train2, y_train2 = preprocess_data(x_train_raw, y_train_raw, 1000)

network = [
    layers.Convolution(kernel_size=11, n_kernels=32, stride=(2,2), build=True, activation="relu"),
    layers.Convolution(kernel_size=7, n_kernels=16, stride=(2,2), build=True, activation="relu"),
    layers.Convolution(kernel_size=2, n_kernels=8, stride=(1,1), build=True, activation="relu"),
    layers.Flatten(),
    layers.Dense(units=100, build=True, activation="relu"),
    layers.Dense(units=10, build=True, activation="softmax")
]

model = Sequential(network, learning_rate=0.1)
model.compile(loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
#%%
import matplotlib.pyplot as plt

x = [i for i in range (3)]
y = model.epoch_losses
#plt.scatter(x,y)

#%%
import dill

with open("cnn_model.pkl", "wb") as arquivo:
    dill.dump(model, arquivo)

#%%
import dill
with open("cnn_model.pkl", "rb") as arquivo:
    model = dill.load(arquivo)

#%%
t = plt.imread('seven.jpg')
t = t.reshape((1,28,28))
t = t.astype(np.float32)


output = jnp.array(model.predict_one(t))
value = output[jnp.argmax(output)][0]
pred = jnp.where(output==value, 1, jnp.where(output!=value, 0, output))

pred = int(jnp.argmax(output))
pred

pred