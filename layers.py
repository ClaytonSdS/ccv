# %%
import numpy as np
from fast import upsampling_gradient as UpSamplingGradient
from fast import FastBatchCorrelate as BatchCorrelate

import jax
import jax.numpy as jnp
from jax import vmap  
from jax.nn import softmax as jnp_softmax
from jax.scipy.signal import correlate

import activations 
from scipy import signal


__all__ = ['Layer','Dense', 'Convolution', 'Tanh', 'Flatten', 'Softmax', 'ReLu', 'Sigmoid']


class Layer():
    def __init__(self, *kwargs):
        self.input = None
        self.output = None
        self.tag = None

        self.islastlayer = False
        self.last_function = None
        self.model_loss = None
        self.activation = {"tanh":activations.Tanh,
                           "softmax":activations.Softmax,
                           "sigmoid":activations.Sigmoid,
                           "relu":activations.ReLu}

    def forward(self):
        pass
    
    def backward(self):
        pass

class Flatten(Layer):
    def __init__(self, input_shape=None, output_shape = None, build = False, *kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.tag = "Flatten"
        self.build = build

    def constructor(self, input_shape):
        if self.output_shape == None:
            self.input_shape = input_shape
            max = jnp.prod(jnp.array(input_shape)) # calculate the product between (x,y,z) = x*y*z
            self.output_shape = (int(max),1)
            return self.output_shape
        
        else:
            return self.output_shape

    def forward(self, input):
        return input.reshape(self.output_shape)

    def backward(self, output, *kwargs):
        return output.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, units, activation, input_size=None, seed=133, build = False, *kwargs):
        super().__init__(*kwargs) 
        # i: input, j: output
        self.key = jax.random.PRNGKey(133) 
        self.key_bias = jax.random.PRNGKey(27)
        self.units = units # output shape (units,1)
        self.tag = "Dense"
        self.build = build

        self.input_size = input_size

        # activation function
        self.activation_function = activation
        self.activation = self.activation[activation]()



    def constructor(self, input_shape):
        self.weights = jax.random.normal(self.key, shape=(self.units, input_shape[0])) #* jnp.sqrt(2. / input_shape)
        self.bias = jax.random.normal(self.key_bias, shape=(self.units, 1)) * 0.1 

        self.input_shape = input_shape
        self.output_shape = self.bias.shape

        # TODO: future validations - possible new fixes will be needed in the future here
        return (self.units, 1)

    def forward(self, input):
        self.input = input
        z = jnp.dot(self.weights, self.input) + self.bias
        return self.activation.forward(z)
    
    def backward(self, output_gradient, learning_rate):        
        if self.islastlayer == "True" and self.activation_function == "softmax" and self.model_loss == "categorical_crossentropy":
            output_gradient = output_gradient

        else:
            output_gradient = self.activation.backward(output_gradient, learning_rate)
        

        biases_gradient = output_gradient
        weights_gradient = jnp.dot(output_gradient, self.input.T)
        
        # Verificação antes de atualizar os pesos
        if jnp.any(jnp.isnan(weights_gradient)) or jnp.any(jnp.isnan(biases_gradient)):
            print("Gradiente de pesos ou viéses contém NaN antes da atualização!")
        
        # Atualizando pesos e bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * biases_gradient

        # Verificação após atualização dos pesos
        if jnp.any(jnp.isnan(self.weights)):
            print("Pesos se tornaram NaN após a atualização!")
            self.weights

        if jnp.any(jnp.isnan(self.bias)):
            print("viéses se tornaram NaN após a atualização!")
            print(self.bias)

        backward = jnp.dot(self.weights.T, output_gradient)
        
        return backward
    

class Convolution(Layer):
    # Conv((32,32,3), kernel_size=2, n_kernels=3)
    def __init__(self, activation, kernel_size, n_kernels, stride, input_shape=None, pad=0, seed=132, build=False, *kwargs):
        super().__init__(*kwargs)

        self.key = jax.random.PRNGKey(10) 
        self.key_bias = jax.random.PRNGKey(173)
        self.tag = "Convolution"
        self.build = build

        self.input_shape = input_shape
        self.pad = pad
        self.stride = stride
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size

        # activation function
        self.activation_function = activation
        self.activation = self.activation[activation]()

    # transforms the output_shape from the previous_layer and pass to the next layer as its input_shape f(output_shape) -> input_shape
    def constructor(self, input_shape, *kwargs):
        input_depth, input_height, input_width = input_shape
        input_height += 2*self.pad
        input_width  += 2*self.pad 

        self.depth = self.n_kernels
        self.input_depth = input_depth
        self.stride_x, self.stride_y = self.stride

        out_shape_x = (input_width + 2 * self.pad - self.kernel_size) // self.stride_x + 1
        out_shape_y = (input_width + 2 * self.pad - self.kernel_size) // self.stride_y + 1

        # (Batch, Channels, Height->rows, Width->columns)  = BCHW
        self.input_shape = (input_depth, input_height, input_width)
        self.output_shape = (self.depth, out_shape_y, out_shape_x)
        self.kernels_shape = (self.depth, input_depth, self.kernel_size, self.kernel_size)

        # setting random values for the biases and kernels.
        self.kernels = jax.random.normal(self.key, shape=self.kernels_shape)
        self.biases = jax.random.normal(self.key_bias, shape=self.output_shape)

        return self.output_shape
                

    def forward(self, input):
        self.input = jnp.array(input)

        self.output = BatchCorrelate(self.input[None, :, :, :], self.kernels, self.stride).result.squeeze(axis=0) 
        self.output += self.biases

        z = self.output
        return self.activation.forward(z) 

    def backward (self, output_gradient, learning_rate):
        if self.islastlayer and self.activation_function == "softmax" and self.model_loss == "categorical_crossentropy":
            output_gradient = output_gradient

        else:
            output_gradient = self.activation.backward(output_gradient, learning_rate)

         # calculating ∂E/∂B
        bias_gradient = output_gradient # -> ∂E/∂B = ∂E/∂Y

        # dilating the ∂E/∂Y matrix and calculating the BatchCorrelate
        print(f"output_gradient shape antes de UpSamplingGradient: {output_gradient.shape}\n")
        output_gradient = UpSamplingGradient(output_gradient, self.stride)

        # calculating ∂E/∂Ki,j

        print(f"input shape: {self.input[:, None, :, :].shape}")
        print(f"output_gradient shape depois de UpSamplingGradient: {output_gradient.shape}\n")

        kernels_gradient = BatchCorrelate(self.input[:, None, :, :], output_gradient[:, None, :, :], (1,1), ).result        

        if self.input_depth != 1:
            pass
            #print("concatenated!")
            #print(f"kernels_gradient antes de concatenate: {kernels_gradient.shape}")
            #kernels_gradient = jnp.concatenate([kernels_gradient[::,::self.depth,::,::], kernels_gradient[::,::-self.depth,::,::]], axis=0)

        print(f"kernels_gradient antes de [:,:,:self.kernel_size, ...] {kernels_gradient.shape}")
        kernels_gradient = kernels_gradient[:,:,:self.kernel_size, :self.kernel_size]
        
        print(f"kernels_gradient depois de [:,:,:self.kernel_size, ...] {kernels_gradient.shape}")
        kernels_gradient = kernels_gradient.reshape(self.depth,self.input_depth,*self.kernels.shape[2:])

        # adding padding
        output_gradient = jnp.pad(output_gradient, pad_width=((0,0),(self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)), mode='constant')
        
        # new implementation so the gradient shape matches the output shape of the previous layer
        if output_gradient.shape[-1] != self.input_shape[-1]:
            dif = output_gradient.shape[-1] - self.input_shape[-1]
            #output_gradient = jnp.pad(output_gradient, pad_width=((0,0),(dif, 0), (dif, 0)), mode='edge')

            print(f"dif: {dif} gradient após reajuste: {output_gradient.shape}")

        kernels_transposed = jnp.transpose(self.kernels, (1, 0, 2, 3))         # Tranposing self.kernels to get: K_i,j to -> K_j,i
        kernels_flipped = jnp.flip(kernels_transposed, axis=(2, 3))            # Flipping the kernel to perfom a true conv operation.

        # calculating ∂E/∂Xj
        input_gradient = BatchCorrelate(output_gradient[None,::,::,::], kernels_flipped, (1,1)).result  # (1,j, inp_size, inp_size)
        print(f"del e, input_gradient BatchCorrelate: {input_gradient.shape}")
        print(f"del e, output_gradient: {output_gradient[None,::,::,::].shape}")
        print(f"del e, kernels_flipped: {kernels_flipped.shape}")
        input_gradient = input_gradient.reshape(*input_gradient.shape[1:])
        print(f"del e, input_gradient: {input_gradient.shape}")

        
        # updating 
        self.kernels -= learning_rate * kernels_gradient
        self.biases  -= learning_rate * bias_gradient


        # returning ∂E/∂X -> input_gradient
        backward = input_gradient
        return backward

    
        

