# Activation Functions
import jax.numpy as jnp
from jax.nn import softmax as jnp_softmax
import numpy as np
import tensorflow as tf

__all__ = ['Tanh', 'Softmax', 'ReLu', 'Sigmoid', 'LeakyRelu']


class Activation():
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.tag = "Activation"
        self.input_shape = None
        self.output_shape = None

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return tf.multiply(output_gradient, self.activation_derivative(self.input))

class LeakyRelu(Activation):
    def __init__(self):
        def leaky_relu(X):
            epsilon = 1e15
            #X = tf.clip_by_value(X, clip_value_min=0.0, clip_value_max=epsilon)
            return tf.nn.leaky_relu(X)

        def leaky_derivative(X):
            with tf.GradientTape() as tape:
                tape.watch(X)
                output = tf.nn.leaky_relu(X) 
            return tape.gradient(output, X)

        super().__init__(leaky_relu, leaky_derivative)

class ReLu(Activation):
    def __init__(self):
        def relu(X):
            epsilon = 1e15
            #X = tf.clip_by_value(X, clip_value_min=0.0, clip_value_max=epsilon)
            return tf.nn.relu(X)

        def relu_derivative(X):
            with tf.GradientTape() as tape:
                tape.watch(X)
                output = tf.nn.relu(X) 
            return tape.gradient(output, X)

        super().__init__(relu, relu_derivative)


class Tanh(Activation):
    def __init__(self):
        def tanh(X):
            return jnp.tanh(X)

        def tanh_derivative(X):
            return 1 - jnp.tanh(X) ** 2

        super().__init__(tanh, tanh_derivative)

class Softmax(Activation):#(Activation):
    def __init__(self):

        def softmax(X):
            #X = tf.clip_by_value(X, -1e8, 1e9)
            softmax_output = tf.nn.softmax(X, axis=1)
            
            print(f"[SOFTMAX] X {X.shape}: {X[0]}")
            #print(f"[SOFTMAX] softmax(X) ({softmax_output.shape}): \n{softmax_output[0]}\n")

            return X
        
        def softmax_derivative(X):
            X_squeezed = tf.squeeze(X, axis=-1)
            #print(f"[SOFTMAX-derivative] X ({X_squeezed.shape}): \n{X_squeezed[0]}\n")
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(X_squeezed)
                output = tf.nn.softmax(X_squeezed, axis=1) 

            grad = tape.gradient(output, X_squeezed)
            grad = grad[::,::,None]
            #print(f"[SOFTMAX-derivative] softmax_gradient(X) ({grad.shape}): \n{grad[0]}")
            return grad

        super().__init__(softmax, softmax_derivative)
        
        
   
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(X):
            return tf.nn.sigmoid(X)

        def sigmoid_derivative(X):
            with tf.GradientTape() as tape:
                tape.watch(X)
                output = tf.nn.sigmoid(X) 
            return tape.gradient(output, X)

        super().__init__(sigmoid, sigmoid_derivative)



