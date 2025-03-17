# %%
import numpy as np

try:
    from .activations import ReLu, Tanh, Sigmoid, Softmax, LeakyRelu
    from .fast import upsampling_gradient as UpSamplingGradient
    from .fast import FastBatchCorrelate as BatchCorrelate

except ImportError:
    from activations import ReLu, Tanh, Sigmoid, Softmax, LeakyRelu
    from fast import upsampling_gradient as UpSamplingGradient
    from fast import FastBatchCorrelate as BatchCorrelate

import tensorflow as tf

import jax
import jax.numpy as jnp
from jax import vmap  
from jax.nn import softmax as jnp_softmax
from jax.scipy.signal import correlate
from jax import lax


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
        self.activation_functions = {"tanh":Tanh,
                           "softmax":Softmax,
                           "sigmoid":Sigmoid,
                           "relu":ReLu,
                           "leaky_relu":LeakyRelu}

    def forward(self):
        pass
    
    def backward(self):
        pass


class BatchNormalization(Layer):
    def __init__(self, epsilon=1e-5, momentum=0.9, forward_verbose=False, backward_verbose=False, *kwargs):
        super().__init__(*kwargs)
        self.epsilon = epsilon  # Pequeno valor para evitar divisão por zero
        self.momentum = momentum  # Fator de momentum para médias móveis
        self.tag = "BatchNorm"

        # Verbosidade
        self.forward_verbose = forward_verbose
        self.backward_verbose = backward_verbose

        # Parâmetros aprendíveis
        self.gamma = None  # Escala
        self.beta = None   # Deslocamento

        # Estatísticas de treinamento
        self.running_mean = None
        self.running_var = None

        # Variáveis intermediárias (usadas no backward)
        self.input = None
        self.batch_mean = None
        self.batch_var = None
        self.normalized_input = None

        # Controle de treinamento vs inferência
        self.training = True  # Inicialmente assume-se que está em treinamento

    def constructor(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.output_shape = input_shape  # Mesma forma que a entrada

        # Inicializar parâmetros aprendíveis
        self.gamma = tf.ones(input_shape[-1])
        self.beta = tf.zeros(input_shape[-1])

        # Inicializar estatísticas de médias móveis
        self.running_mean = tf.zeros(input_shape[-1])
        self.running_var = tf.ones(input_shape[-1])

        return self.output_shape

    def forward(self, input, batch_size):
        self.input = input

        if self.forward_verbose:
            print(f"[BatchNorm{self.pos_in_model}-forward] Input shape: {input.shape}")

        
        # Calcular médias e variâncias do batch manualmente
        self.batch_mean = tf.reduce_mean(input, axis=0)
        self.batch_var = tf.reduce_mean((input - self.batch_mean) ** 2, axis=0)

        # Normalizar a entrada
        self.normalized_input = (input - self.batch_mean) / tf.sqrt(self.batch_var + self.epsilon)

        # Atualizar médias móveis
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var

        #self.normalized_input = (input - self.running_mean) / tf.sqrt(self.running_var + self.epsilon)

        # Escalar e deslocar
        self.output = self.gamma * self.normalized_input + self.beta


        if self.forward_verbose:
            print(f"[BatchNorm{self.pos_in_model}-forward] Output shape: {self.output.shape}")

        return self.output

    def backward(self, gradient, learning_rate):
        if self.backward_verbose:
            print(f"[BatchNorm{self.pos_in_model}-backward] Gradient shape: {gradient.shape}")

        # Gradientes em relação a gamma e beta
        gamma_gradient = tf.reduce_sum(gradient * self.normalized_input, axis=0)
        beta_gradient = tf.reduce_sum(gradient, axis=0)

        # Gradiente em relação à entrada normalizada
        normalized_input_gradient = gradient * self.gamma

        # Gradiente em relação à entrada
        batch_size = tf.cast(tf.shape(self.input)[0], tf.float32)
        input_gradient = (1. / batch_size) * (1. / tf.sqrt(self.batch_var + self.epsilon)) * (
            batch_size * normalized_input_gradient 
            - tf.reduce_sum(normalized_input_gradient, axis=0)
            - self.normalized_input * tf.reduce_sum(normalized_input_gradient * self.normalized_input, axis=0)
        )

        # Atualizar parâmetros aprendíveis
        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient

        if tf.reduce_any(tf.math.is_nan(input_gradient)).numpy():
            print(f"[BatchNorm{self.pos_in_model}-backward] NaN detected in input gradient")

        if self.backward_verbose:
            print(f"[BatchNorm{self.pos_in_model}-backward] Updated gamma: {self.gamma.shape}, beta: {self.beta.shape}")

        return input_gradient

    def set_training(self, is_training):
        # Método para definir explicitamente se está em modo de treinamento ou inferência
        self.training = is_training

class Flatten(Layer):
    def __init__(self, input_shape=None, output_shape = None, forward_verbose = False, backward_verbose = False, *kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.tag = "Flatten"
            
        # setting verboses
        self.forward_verbose = forward_verbose
        self.backward_verbose = backward_verbose

    def constructor(self, input_shape, batch_size):
        self.input_shape = input_shape
        batch = batch_size
        max = jnp.prod(jnp.array(input_shape[1:])) # calculate the product between (x,y,z) = x*y*z
        self.output_shape = (batch, int(max), 1)
        return self.output_shape
    

    def forward(self, input, batch_size):
        self.input_shape = input.shape
        batch = batch_size
        max = tf.reduce_prod(input.shape[1:])  # Calcula o produto entre (batch, x, y, z) = x * y * z
        self.output_shape = (batch, int(max), 1)
        self.output = tf.reshape(input, self.output_shape)  # Use tf.reshape em vez de input.reshape

        # nan in the array
        if tf.reduce_any(tf.math.is_nan(self.output)).numpy():
            print(f"[FLATTEN{self.pos_in_model}-forward] nan found output:{self.output.shape}")
            

        else:
            if self.forward_verbose:
                print(f"[FLATTEN{self.pos_in_model}-forward] input shape: {input.shape} | output_shape: {self.output.shape}")
            return self.output

    def backward(self, output, *kwargs):
        backward = tf.reshape(output, self.input_shape)

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
        self.activation = self.activation_functions [activation]()



    def constructor(self, input_shape, batch_size):
        input_shape = list(input_shape)

        # checking if the last axis is 1 and "pseudo-squeezing it"
        if input_shape[-1] == 1:
            input_shape = tuple(input_shape[:-1])

        batch = batch_size
        in_neurons = input_shape[-1]
        out_neurons = self.units

        initializer = tf.keras.initializers.HeNormal()
        self.weights =  initializer(shape=(in_neurons, out_neurons))
        self.bias = tf.zeros(shape=(out_neurons, 1))

        self.input_shape = input_shape
        self.output_shape = (batch, *self.bias.shape)

        # TODO: future validations - possible new fixes will be needed in the future here
        return self.output_shape

    def forward(self, input, batch):
        self.input = tf.squeeze(input, axis=-1)
        self.input = tf.transpose(self.input)
        self.weights = tf.transpose(self.weights)

        
        try:
            z = tf.linalg.matmul(self.weights, self.input) # dot product between weights and input
            z += self.bias

            # input nan
            if tf.reduce_any(tf.math.is_nan(self.input)).numpy():   
                print(f"input_nan : True")
            # weights nan
            if tf.reduce_any(tf.math.is_nan(self.weights)).numpy():   
                print(f"weights_nan : True")

                # bias nan
            if tf.reduce_any(tf.math.is_nan(self.bias)).numpy():   
                print(f"bias_nan : True")

            z = tf.transpose(z)
            z = z[..., None]
            z = self.activation.forward(z)

            # nan in the array
            if tf.reduce_any(tf.math.is_nan(z)).numpy():
                print(f"[DENSE{self.pos_in_model}-forward] nan found output:{z.shape}")
                print(z.shape)
                return float('nan')
            
            else:
                if self.forward_verbose:
                    print(f"[DENSE{self.pos_in_model}-forward]  weights: {self.weights.shape} | input: {self.input.shape}  | bias: {self.bias.shape} | z: {z.shape} ")
                return z

        except tf.errors.InvalidArgumentError as error:
            if self.weights.shape[0] == self.input.shape[0]:
                self.weights = tf.transpose(self.weights)        # transposing weights again to have (out_neurons, in_neurons)
                z = tf.linalg.matmul(self.weights, self.input)   # dot product between weights and input
                z += self.bias

                # input nan
                if tf.reduce_any(tf.math.is_nan(self.input)).numpy():   
                    print(f"(e) input_nan: True")
                # weights nan
                if tf.reduce_any(tf.math.is_nan(self.weights)).numpy():   
                    print(f"(e) weights_nan: True")

                # bias nan
                if tf.reduce_any(tf.math.is_nan(self.bias)).numpy():   
                    print(f"(e) bias_nan: True")

                z = tf.transpose(z)
                z = z[..., None]
                z = self.activation.forward(z)

                # nan in the array
                if tf.reduce_any(tf.math.is_nan(z)).numpy():
                    print(f"[DENSE{self.pos_in_model}-forward] (e) nan true after activation(z):{z.shape}")
                    return float('nan')
                
                else:
                    return z


            else:
                print(f"O erro em DENSE{self.pos_in_model} persiste / weights: {self.weights.shape} | input: {self.input.shape}  | bias: {self.bias.shape}")
                return float('nan')
        
        
    
    def backward(self, output_gradient, learning_rate): 
        #output_gradient = self.activation.backward(output_gradient, learning_rate)
        #output_gradient = tf.squeeze(output_gradient, axis=-1)
            
        if self.islastlayer and self.activation_function == "softmax" and self.model_loss == "categorical_crossentropy":
            output_gradient = output_gradient
            #output_gradient = self.activation.backward(output_gradient, learning_rate)
            output_gradient = tf.squeeze(output_gradient, axis=-1)
            
        else:
            
            output_gradient = self.activation.backward(output_gradient, learning_rate)
            output_gradient = tf.squeeze(output_gradient, axis=-1)

            
        if self.backward_verbose:
            print(f"[DENSE{self.pos_in_model}-backward] bias: {self.bias.shape} activation(output_gradient): {output_gradient.shape}")

        biases_gradient = tf.reduce_sum(output_gradient, axis=0, keepdims=True)
        biases_gradient = tf.transpose(biases_gradient)

        # tranposing ∂E/∂Y and input, to match the shape for the matrix multiplication
        output_gradient = tf.transpose(output_gradient)
        self.input = tf.transpose(self.input)
        weights_gradient = tf.linalg.matmul(output_gradient, self.input)

        if self.backward_verbose:
            print(f"[DENSE{self.pos_in_model}-backward] grad.T: {output_gradient.shape} | input.T: {self.input.shape}| bias_grad: {biases_gradient.shape} | weights_grad:{weights_gradient.shape}")
        
        # updating parameters: weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * biases_gradient

        if self.backward_verbose:
            print(f"[DENSE{self.pos_in_model}-backward] Updated: weights {self.weights.shape} | bias: {self.bias.shape}")    

        # ///////////////
        backward = tf.linalg.matmul(tf.transpose(self.weights), output_gradient)
        backward = tf.transpose(backward)[..., None]

        
        if self.backward_verbose:
            print(f"[DENSE{self.pos_in_model}-backward] del_E/del_X:{backward.shape}")

        # nan check:
        if tf.reduce_any(tf.math.is_nan(backward)).numpy():
            print(f"[DENSE{self.pos_in_model}-backward] nan found inp_grad:{backward.shape}")
            return float('nan')
        
        else:
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
        self.activation = self.activation_functions[activation]()

    # transforms the output_shape from the previous_layer and pass to the next layer as its input_shape f(output_shape) -> input_shape
    def constructor(self, input_shape, *kwargs):
        #batch, input_depth, input_height, input_width = input_shape            # old approach
        batch, self.input_height, self.input_width, self.input_depth = input_shape             # a given input shape looks like [batch, in_height, in_width, in_channels]
        self.input_height += 2*self.pad
        self.input_width  += 2*self.pad 
        self.input_shape = input_shape

        self.out_channels = self.n_kernels                                            # out_channels = depth (i) = n_kernels
        self.in_channels = self.input_depth                                                # in_channels  = input_depth (j) = n_channels of the input
        self.stride_x, self.stride_y = self.stride                                    # spliting stride(x,y) in s_x and s_y

        out_shape_x = (self.input_width + 2 * self.pad - self.kernel_size) // self.stride_x + 1          # out shape equation on x-direction
        out_shape_y = (self.input_height + 2 * self.pad - self.kernel_size) // self.stride_y + 1          # out shape equation on y-direction
        
        #self.output_shape = (batch, self.depth, out_shape_y, out_shape_x)   # old approach
        self.output_shape = (batch, out_shape_y, out_shape_x, self.out_channels)                              # The output shape will be equal to:   [batch, out_shape_y, out_shape_x, out_channels]
        self.kernels_shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)        # kernel shape looks like:             [filter_height, filter_width, in_channels, out_channels]
        self.biases_shape = (out_shape_y, out_shape_x, self.out_channels)

        # setting random values for the biases and kernels.
        initializer = tf.keras.initializers.HeNormal()
        self.kernels = initializer(shape=self.kernels_shape)
        self.biases = tf.zeros(shape=self.biases_shape)
        
        return self.output_shape
                

    def forward(self, input, batch):
        self.input = input

        try:
            if self.forward_verbose:
                print(f"[CONV{self.pos_in_model}-forward] input shape: {input.shape} | kernels: {self.kernels.shape} | bias: {self.biases.shape}")

            self.output = tf.nn.convolution(input, self.kernels, self.stride, "VALID")
            self.output += self.biases

            if self.forward_verbose:
                print(f"[CONV{self.pos_in_model}-forward] input shape: {input.shape} | kernels: {self.kernels.shape} | output (cross_correlation): {self.output.shape} | bias: {self.biases.shape}")

            if tf.reduce_any(tf.math.is_nan(self.output)).numpy():
                print(f"[CONV{self.pos_in_model}-forward] nan found in output::{self.output.shape}")
                print(self.output)
                return float('nan')
            
            else:
                return self.activation.forward(self.output)

        except TypeError:
            return float('nan')

        
        

    def backward (self, output_gradient, learning_rate):

        self.activation.backward(output_gradient, learning_rate)

        """
        OLD CODE
        if self.islastlayer and self.activation_function == "softmax" and self.model_loss == "categorical_crossentropy":
            output_gradient = output_gradient
            #output_gradient = self.activation.backward(output_gradient, learning_rate)

        else:
            output_gradient = self.activation.backward(output_gradient, learning_rate)
            """


         # calculating ∂E/∂B (biases_gradient)
        bias_gradient = tf.reduce_sum(output_gradient, axis=0) # -> ∂E/∂B = ∂E/∂Y

        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] output_gradient: {output_gradient.shape} | biases: {self.biases.shape} | bias_grad: {bias_gradient.shape}")
        

        # dilating the ∂E/∂Y tensor
        output_gradient = UpSamplingGradient(output_gradient, self.stride)

        # calculating ∂E/∂K (kernels_gradient)
        self.input = tf.transpose(self.input, [3,1,2,0])
        output_gradient = tf.transpose(output_gradient, [2,1,0,3])
        kernels_gradient = tf.nn.convolution(self.input, output_gradient, (1,1), "VALID")
        kernels_gradient = tf.transpose(kernels_gradient, [2,1,0,3 ])        # Tranposing to match kernel shape as  (out_channels, in_channels, filter_height, filter_width) 
        
        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] kernels_grad(input, output_gradient) input: {self.input.shape} | output_gradient: {output_gradient.shape} | kernels_gradient: {kernels_gradient.shape} | kernels: {self.kernels.shape} ")
        
        
        # adding padding to perfom a full cross-correlation between (∂E/∂Y, rot_180(K))
        output_gradient = tf.pad(output_gradient,
                                paddings=[[self.kernel_size - 1, self.kernel_size - 1], 
                                          [self.kernel_size - 1, self.kernel_size - 1],
                                          [0, 0], 
                                          [0, 0]], mode='CONSTANT')
        output_gradient = tf.transpose(output_gradient, [2,1,0,3])        # Tranposing (height, width, batch, num_kernels) to (batch, height, width, num_kernels)
        kernels_flipped =  tf.reverse(self.kernels, axis=[0, 1])          # Flipping the kernel to perfom a true conv operation.
        kernels_flipped = tf.transpose(kernels_flipped, [0,1,3,2])        # Tranposing (filter_height, filter_width, in_channels, out_channels) to (filter_height, filter_width, out_channels, in_channels)

        # calculating ∂E/∂Xj
        input_gradient = tf.nn.convolution(output_gradient, kernels_flipped, (1,1), "VALID")


        if self.backward_verbose:
            print(f"[CONV{self.pos_in_model}-backward] input_grad(output_gradient, kernels_flip) output_gradient: {output_gradient.shape} | kernels_flipped: {kernels_flipped.shape} | input_grad:{input_gradient.shape}")


        # updating parameters: kernels and biases 
        self.kernels -= learning_rate * kernels_gradient
        self.biases  -= learning_rate * bias_gradient

        # nan check
        if tf.reduce_any(tf.math.is_nan(input_gradient)).numpy():
            print(f"[CONV{self.pos_in_model}-backward] nan found in inp_grad::{input_gradient.shape}")
            return float('nan')

        else:
            # returning ∂E/∂X -> input_gradient 
            return input_gradient

    
        

