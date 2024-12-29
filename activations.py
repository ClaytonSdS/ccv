# Activation Functions
import jax.numpy as jnp
from jax.nn import softmax as jnp_softmax
import numpy as np

__all__ = ['Tanh', 'Softmax', 'ReLu', 'Sigmoid']


class Activation():
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.tag = "Activation"
        self.input_shape = None
        self.output_shape = None
        self.y_true = None

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate, y_true=None):
        return jnp.multiply(output_gradient, self.activation_derivative(self.input, y_true))


class Tanh(Activation):
    def __init__(self):
        def tanh(X):
            return jnp.tanh(X)

        def tanh_derivative(X):
            return 1 - jnp.tanh(X) ** 2

        super().__init__(tanh, tanh_derivative)

class Softmax(Activation):#(Activation):
    def __init__(self):
        self.y_true = None

        def softmax(X, axis=1):
            epsilon = 1e-7
            softmax_output = jnp_softmax(X, axis)
            return softmax_output
        

        def softmax_derivative(X, y_true):
            s = jnp_softmax(X, axis=1)          # Calcula o softmax
            grad = s - y_true 
            return grad

        super().__init__(softmax, softmax_derivative)
        
        
   
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))

        def sigmoid_prime(x, y_true):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class ReLu(Activation):
    def __init__(self):
        def relu(X):
            #print(f"Activation | ReLu Layer [{X.shape}]: X = {X[0]}\nrelu(X)={jnp.maximum(0,X)[0]}")
            return jnp.maximum(0,X)

        def relu_derivative(X, y_true):
            epsilon = 1e-10 
            return jnp.where(X>=0, 1, jnp.where(X<0, 0, X))

        super().__init__(relu, relu_derivative)


