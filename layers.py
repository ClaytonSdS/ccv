# %%
import numpy as np

try:
    from .fast import upsampling_gradient as UpSamplingGradient
    from .fast import FastBatchCorrelate as BatchCorrelate

except ImportError:
    from fast import upsampling_gradient as UpSamplingGradient
    from fast import FastBatchCorrelate as BatchCorrelate

import tensorflow as tf

import jax
import jax.numpy as jnp
from jax import vmap  
from jax.nn import softmax as jnp_softmax
from jax.scipy.signal import correlate
from jax import lax

import activations 
from scipy import signal


__all__ = ['Layer','Dense', 'Convolution', 'Tanh', 'Flatten', 'Softmax', 'ReLu', 'Sigmoid']


class Layer():
    def __init__(self, *kwargs):
        self.input = None
        self.output = None
        self.tag = None

        self.Y_BATCH = None

        self.forward_verbose = None
        self.backward_verbose = None

        self.pos_in_model = None

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
    def __init__(self, input_shape=None, output_shape = None, forward_verbose = False, backward_verbose = False, *kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.tag = "Flatten"

        # setting verboses
        self.forward_verbose = forward_verbose
        self.backward_verbose = backward_verbose

    def constructor(self, input_shape):
        if self.output_shape == None:
            self.input_shape = input_shape
            batch = self.input_shape[0]
            max = jnp.prod(jnp.array(input_shape[1:])) # calculate the product between (x,y,z) = x*y*z
            self.output_shape = (batch, int(max),1)
            return self.output_shape
        
        else:
            return self.output_shape

    def forward(self, input):
        output = input.reshape(self.output_shape)
        if self.forward_verbose:
            print(f"[FLATTEN{self.pos_in_model}-forward] input shape: {input.shape} | output_shape: {output.shape}")
        return output

    def backward(self, output, *kwargs):
        backward = output.reshape(self.input_shape)

        if jnp.any(jnp.isnan(backward)):
            if self.backward_verbose:
                print(f"[FLATTEN{self.pos_in_model}-backward] NAN WARNING del_E/del_X:{jnp.sum(backward)} del_E/del_X:{backward.shape}")

            return "nan"
        
        else:
            if self.backward_verbose:
                print(f"[FLATTEN{self.pos_in_model}-backward] del_E/del_X:{backward.shape}")

            return backward

class Dense(Layer):
    def __init__(self, units, activation, input_size=None, weight_seed=1, bias_seed=2, forward_verbose = False, backward_verbose = False, *kwargs):
        super().__init__(*kwargs) 
        # i: input, j: output
        self.key_weights = jax.random.PRNGKey(weight_seed) 
        self.key_bias = jax.random.PRNGKey(bias_seed)
        self.units = units # output shape (units,1)
        self.tag = "Dense"

        # setting verboses
        self.forward_verbose = forward_verbose
        self.backward_verbose = backward_verbose

        self.input_size = input_size

        # activation function
        self.activation_function = activation
        self.activation = self.activation[activation]()



    def constructor(self, input_shape):
        batch = input_shape[0]
        self.weights = jax.random.normal(self.key_weights, shape=(self.units, input_shape[1])) #* jnp.sqrt(2. / input_shape)
        self.bias = jax.random.normal(self.key_bias, shape=(self.units, 1)) * 0.1 

        self.input_shape = input_shape
        self.output_shape = (batch, *self.bias.shape)

        # TODO: future validations - possible new fixes will be needed in the future here
        return self.output_shape

    def forward(self, input):
        self.input = jnp.squeeze(input, axis=-1).T

        try:
            z = jnp.dot(self.weights, self.input) + self.bias
            z = z.T[..., None]

        except TypeError:
            print(f"[DENSE{self.pos_in_model}-forward] SHAPE ERROR: weights: {self.weights.shape} | input: {self.input.shape} | bias:{self.bias.shape}")

        
        if self.forward_verbose:
            print(f"[DENSE{self.pos_in_model}-forward]input: output(z): {z.shape} | activation(z):{self.activation.forward(z).shape}")

        return self.activation.forward(z)
    
    def backward(self, output_gradient, learning_rate):     

        if self.islastlayer and self.activation_function == "softmax" and self.model_loss == "categorical_crossentropy":
            output_gradient = output_gradient
            output_gradient = jnp.squeeze(output_gradient, axis=-1)   
            
            if self.backward_verbose:
                print(f"[DENSE{self.pos_in_model}-backward] Not using activation")
            #output_gradient = jnp.squeeze(output_gradient, axis=-1)   

        else:
            output_gradient = self.activation.backward(output_gradient, learning_rate)
            output_gradient = jnp.squeeze(output_gradient, axis=-1)   
        
        biases_gradient = jnp.sum(output_gradient, axis=0, keepdims=True).T
        output_gradient = output_gradient.T
        weights_gradient = jnp.dot(output_gradient, self.input.T)

        if self.backward_verbose:
            print(f"[DENSE{self.pos_in_model}-backward] grad.T: {output_gradient.shape} | input.T: {self.input.T.shape}| bias_grad: {biases_gradient.shape} | weights_grad:{weights_gradient.shape}")
        
        
        # updating 
        if self.backward_verbose:
            print(f"[DENSE{self.pos_in_model}-backward] updating weights and biases: weights {self.weights.shape} | w.grad: {weights_gradient.shape} / bias: {self.bias.shape} | b.grad: {biases_gradient.shape}")    
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * biases_gradient

        if self.backward_verbose:
            print(f"[DENSE{self.pos_in_model}-backward] updated weights and biases: weights {self.weights.shape} / bias: {self.bias.shape}")    

        # (100,10) (32,10)
        backward = jnp.dot(self.weights.T, output_gradient)
        backward = backward.T[..., None]


        if jnp.any(jnp.isnan(backward)):
            if self.backward_verbose:
                print(f"[DENSE{self.pos_in_model}-backward] NAN WARNING del_E/del_X:{jnp.sum(backward)} del_E/del_X:{backward.shape}")

            return "nan"
        
        else:
            
            if self.backward_verbose:
                print(f"[DENSE{self.pos_in_model}-backward] del_E/del_X:{backward.shape}")
            return backward

        
        
    

class Convolution(Layer):
    # Conv((32,32,3), kernel_size=2, n_kernels=3)
    def __init__(self, activation, kernel_size, n_kernels, stride, input_shape=None, pad=0, kernels_seed=3, bias_seed=4, forward_verbose = False, backward_verbose = False, *kwargs):
        super().__init__(*kwargs)

        self.key_kernels = jax.random.PRNGKey(kernels_seed) 
        self.key_bias = jax.random.PRNGKey(bias_seed)
        self.tag = "Convolution"
        
        # setting verboses
        self.forward_verbose = forward_verbose
        self.backward_verbose = backward_verbose

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
        batch, input_depth, input_height, input_width = input_shape
        input_height += 2*self.pad
        input_width  += 2*self.pad 

        self.depth = self.n_kernels
        self.input_depth = input_depth
        self.stride_x, self.stride_y = self.stride

        out_shape_x = (input_width + 2 * self.pad - self.kernel_size) // self.stride_x + 1
        out_shape_y = (input_width + 2 * self.pad - self.kernel_size) // self.stride_y + 1

        # (Batch, Channels, Height->rows, Width->columns)  = BCHW
        self.input_shape = (batch, input_depth, input_height, input_width)
        self.output_shape = (batch, self.depth, out_shape_y, out_shape_x)
        self.kernels_shape = (self.depth, input_depth, self.kernel_size, self.kernel_size)

        # setting random values for the biases and kernels.
        self.kernels = jax.random.normal(self.key_kernels, shape=self.kernels_shape)
        self.biases = jax.random.normal(self.key_bias, shape=self.output_shape)

        return self.output_shape
                

    def forward(self, input):
        self.input = jnp.array(input)
        #self.output = BatchCorrelate(self.input, self.kernels, self.stride).result.squeeze(axis=0) 

        try:
            self.output = BatchCorrelate(self.input, self.kernels, self.stride).result
            self.output += self.biases
            z = self.activation.forward(self.output)
            
            if self.forward_verbose:
                print(f"[CONV{self.pos_in_model}-forward] input shape: {self.input.shape} | kernels: {self.kernels.shape} | output (cross_correlation): {self.output.shape} | bias: {self.biases.shape} | activation: {z.shape}")
            
            return z

        except TypeError:
            print(f"[CONV{self.pos_in_model}-forward] SHAPE ERROR: input shape: {self.input.shape} | kernels: {self.kernels.shape}")
        
            return BatchCorrelate(self.input, self.kernels, self.stride).result

    def backward (self, output_gradient, learning_rate):
        if self.islastlayer and self.activation_function == "softmax" and self.model_loss == "categorical_crossentropy":
            output_gradient = output_gradient
            #output_gradient = self.activation.backward(output_gradient, learning_rate)

        else:
            output_gradient = self.activation.backward(output_gradient, learning_rate)

         # calculating ∂E/∂B
        bias_gradient = output_gradient # -> ∂E/∂B = ∂E/∂Y

        # dilating the ∂E/∂Y matrix and calculating the BatchCorrelate
        output_gradient = UpSamplingGradient(output_gradient, self.stride)

        # calculating ∂E/∂Ki,j
        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] input: {self.input.shape} | UpSampler(output_gradient): {output_gradient.shape} | kernels: {self.kernels.shape}")
        

        feature_groups = int(self.input_depth / self.n_kernels)
        # Checking the input_size in contrast with the num. of kernels to match feature_group_count = 1
        if self.input_depth != self.n_kernels and self.pos_in_model >= 1e15:
            raise ValueError(f"Feature dimension mismatch: Expected number of kernels be {self.input_depth}, but got {self.n_kernels} in {self.tag} layer {self.pos_in_model}")  
        
        """
        previous code: 
        kernels_gradient = BatchCorrelate(self.input, output_gradient, (1,1), feature_groups).result  

        JAX:
        input_shape = (batch, 1in_channels, 2in_height, 3in_width)
        kernels_shape = (0out_channels, 1in_channels, 2self.kernel_size, 3self.kernel_size)

        TENSORFLOW:
        input [batch, in_height, in_width, in_channels]
        kernel [filter_height, filter_width, in_channels, out_channels]
        """

        inp = jnp.transpose(self.input, (1, 3, 2, 0))    
        out = jnp.transpose(output_gradient, (2, 3, 0, 1)) 
        kernels_gradient = jnp.array(tf.nn.convolution(inp, out, (1,1), "VALID"))
        kernels_gradient = jnp.transpose(kernels_gradient, (3,0,2,1))           # Tranposing to match kernel shape as  (out_channels, in_channels, filter_height, filter_width) 

        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] inp: {inp.shape} | out: {out.shape} | kernels_gradient.T: {kernels_gradient.shape}")      

        
        # adding padding
        output_gradient = jnp.pad(output_gradient, pad_width=((0,0),(0,0),(self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)), mode='constant')
        
        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] Pad(output_gradient): {output_gradient.shape}")    

        """
        previous code: 
        kernels_transposed = jnp.transpose(self.kernels, (1, 0, 2, 3))         # Tranposing self.kernels to get: K_i,j to -> K_j,i
        kernels_flipped = jnp.flip(kernels_transposed, axis=(2, 3))            # Flipping the kernel to perfom a true conv operation.

        """
        kernels_flipped = jnp.flip(self.kernels, axis=(2, 3))            # Flipping the kernel to perfom a true conv operation.
        kernels_flipped = jnp.transpose(kernels_flipped, (2,3,0,1))      # Tranposing to match its shape as (filter_height, filter_width, in_channels, out_channels) 
        # 2,2,16,64

        output_gradient = jnp.transpose(output_gradient, (0,3,2,1))

        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] output_gradient.T: {output_gradient.shape} | kernels_flipped.T: {kernels_flipped.shape}")    
        
        # calculating ∂E/∂Xj
        input_gradient = jnp.array(tf.nn.convolution(output_gradient, kernels_flipped, (1,1), "VALID"))
        #input_gradient = jnp.transpose(jnp.array(input_gradient), (0,3,2,1))
        #input_gradient = BatchCorrelate(output_gradient, kernels_flipped, (1,1)).result  # (1,j, inp_size, inp_size)
        

        # updating 
        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] updating kernels and biases: kernels {self.kernels.shape} | k.grad: {kernels_gradient.shape} / bias: {self.biases.shape} | b.grad: {bias_gradient.shape}")    
        
        self.kernels -= learning_rate * kernels_gradient
        self.biases  -= learning_rate * bias_gradient

        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] del_E/del_X:{input_gradient.shape}")

        # returning ∂E/∂X -> input_gradient
        backward = jnp.transpose(input_gradient, (0,3,2,1)) # # Tranposing to match: (batch, in_channels, in_height, in_width) 


        if jnp.any(jnp.isnan(backward)):
            if self.backward_verbose:
                print(f"[CONV{self.pos_in_model}-backward] NAN WARNING del_E/del_X:{jnp.sum(backward)} del_E/del_X:{backward.shape}")

            return "nan"
        
        else:
            if self.backward_verbose:
                print(f"[CONV{self.pos_in_model}-backward] del_E/del_X:{backward.shape}")
                
            return backward

    
        

