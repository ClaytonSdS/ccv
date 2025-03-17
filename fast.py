# %%
import jax.numpy as jnp
from jax import vmap  
from jax.lax import dynamic_slice
from jax import lax 
import jax.scipy.signal as jax_signal
import jax.random
import tensorflow as tf

__all__ = ['upsampling_gradient', 'FastBatchCorrelate', 'FastBatchCorrelate']



class FastBatchCorrelate:
    def __init__(self, input, kernel, stride, feature_group=1, mode="VALID"):
        stride = stride[::-1]
        self.result = jax.lax.conv_general_dilated(
            lhs=input,
            rhs=kernel,
            window_strides=stride,
            padding=mode,
            feature_group_count=feature_group
            #dimension_numbers=("NCHW", "OIHW", "NCHW")
        )

class FastBatchConvolve(FastBatchCorrelate):
    def __init__(self, input, kernel, stride, mode="VALID"):
        kernel_width = kernel.shape[3] - 1
        kernel_height = kernel.shape[2] - 1
        kernel = jnp.flip(kernel, axis=(2, 3))
        input =  jnp.pad(input, pad_width=((0,0),(0,0),(kernel_height, kernel_height), (kernel_width, kernel_width)), mode='constant')
        
        super().__init__(input, kernel, stride, mode)


def upsampling_gradient(output_gradient, stride):
    # Breaking output_gradient shape, i.e., breaking: [batch_size, height, width, channels] 
    batch_size, height, width, channels = output_gradient.shape

    # Original gradient shape
    original_shape = tf.shape(output_gradient)

    # Breaking stride for upsampling
    scale_y, scale_x = stride
    
    # Usando tf.image.resize para aumentar a resolução do gradiente
    upsampled_gradient = tf.image.resize(output_gradient, 
                                         [height * scale_y, width * scale_x], 
                                         method='nearest')  
    
    # Convert to JAX array for using JAX slicing function .at[...].set(...)
    upsampled_gradient = jnp.array(upsampled_gradient)              
    false = jnp.zeros(original_shape, dtype=bool)           # Array filled with False values, serving as a flag
    flag_value = False                                      # The value to search for in the dilated array, i.e, upsampled_gradient
    
    # Flagging specific values that need to be maintained (according to stride), i.e, x1, x2, ...
    upsampled_gradient = upsampled_gradient.at[::,::scale_x, ::scale_y,::].set(false)           
    
    """
    Example:
    stride = (2,2), output_gradient = array([[...[[ x1 ,  x2],
                                                  [ x3 ,  x4]],...]], dtype=float32)>


    After upsampling using tensorflow.image.resize:
    upsampled_gradient = tf.image.resize(output_gradient, [height * scale_y, width * scale_x], method='nearest')  
    upsampled_gradient = array([[...[[ x1,     x1,   x2 ,     x2],
                                     [ x1,     x1,   x2,      x2],                        -> =  (stride - 1)
                                     [ x3 ,    x3,   x4 ,     x4],
                                     [ x3,     x3,   x4,      x4]]...]], dtype=float32    -> =  (stride - 1)
                                                ↑              ↑
                                            (stride-1)     (stride-1)

    After flagging x1, x2, ...
    upsampled_gradient = upsampled_gradient.at[::,::scale_x, ::scale_y,::].set(false)
    upsampled_gradient = array([[...[[ False,     x1,   False,     x2],
                                     [ x1,        x1,   x2,        x2],
                                     [ False ,    x3,   False,     x4],
                                     [ x3,        x3,   x4,        x4]]...]], dtype=float32)
    """

    # Find coords where the value is not False and set them to zero
    zero_coords = jnp.where(upsampled_gradient != flag_value)                          
    upsampled_gradient = upsampled_gradient.at[zero_coords].set(0)      # set the all the non-flagged values to zero

    # Restoring the original gradient values back to the their positions before the upsampling process
    upsampled_gradient = upsampled_gradient.at[::,::scale_x, ::scale_y,::].set(output_gradient)    

    """
    Following the above example, what we get in this step is:
    zero_coords = jnp.where(upsampled_gradient != flag_value) 
    zero_coords = (Array([ 0,  0,  0, ..., batch-1, batch-1, batch-1], dtype=int32), 
                   Array([0, 0, 0, ..., 3, 3, 3], dtype=int32), 
                   Array([1, 1, 1, ..., 3, 3, 3], dtype=int32), 
                   Array([ 0,  1,  2, ..., in_channels-3, in_channels-2, in_channels-1], dtype=int32))


    Setting zero at the coordinates obtained from zero_coords in the upsampled_gradient array, i.e., replacing the False values:                                  
    upsampled_gradient = upsampled_gradient.at[zero_coords].set(0)
    upsampled_gradient = array([[...[[ False,    0,   False,    0],
                                     [ 0,        0,   0,        0],
                                     [ False ,   0,   False,    0],
                                     [ 0,        0,   0,        0]]...]],  dtype=float32)

    Finally, restoring the original values x1, x2, x3, x4, etc... to their respective positions in the upsampled_gradient array:

    upsampled_gradient = upsampled_gradient.at[::,::scale_x, ::scale_y,::].set(grad)
    upsampled_gradient = array([[...[[ x1,    0,   x2,   0],
                                     [ 0,     0,   0,    0],  
                                     [ x3 ,   0,   x4,   0],
                                     [ 0,     0,   0,    0]]...]], dtype=float32)

    """
    
    gradient = tf.convert_to_tensor(upsampled_gradient)                         # converting it back to tensorflow data type
    return gradient

