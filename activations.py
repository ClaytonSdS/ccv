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

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return jnp.multiply(output_gradient, self.activation_derivative(self.input))


class Tanh(Activation):
    def __init__(self):
        def tanh(X):
            return jnp.tanh(X)

        def tanh_derivative(X):
            return 1 - jnp.tanh(X) ** 2

        super().__init__(tanh, tanh_derivative)

class Softmax(Activation):
    def __init__(self):

        def softmax(X, axis=0):
            epsilon = 1e-7
            #X = jnp.where(X==0, epsilon, jnp.where(X!=0, X, X))
            #print(f"X: {X}")
            X_stable = X - jnp.max(X, axis=axis, keepdims=True)
            softmax_output = jnp_softmax(X, axis)
            return softmax_output
        

        def softmax_derivative(X):
            size = jnp.size(X)
            self.output = jnp_softmax(X, 0)
            return jnp.dot(jnp.tile(self.output, size) * (jnp.identity(size) - jnp.transpose(jnp.tile(self.output, size))), X)

        super().__init__(softmax, softmax_derivative)
        
   
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class ReLu(Activation):
    def __init__(self):
        def relu(X):
            return jnp.maximum(0,X)

        def relu_derivative(X):
            epsilon = 1e-10 
            return jnp.where(X>=0, 1, jnp.where(X<0, epsilon, X))

        super().__init__(relu, relu_derivative)


