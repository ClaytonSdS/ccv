# %%
import jax.numpy as jnp
from jax import vmap  
from jax.lax import dynamic_slice
from jax import lax 
import jax.scipy.signal as jax_signal
import jax.random

__all__ = ['upsampling_gradient', 'FastBatchCorrelate', 'FastBatchCorrelate']



class FastBatchCorrelate:
    def __init__(self, input, kernel, stride, mode="VALID"):
        stride = stride[::-1]
        self.result = jax.lax.conv_general_dilated(
            lhs=input,
            rhs=kernel,
            window_strides=stride,
            padding=mode,
            #dimension_numbers=("NCHW", "OIHW", "NCHW")
        )

class FastBatchConvolve(FastBatchCorrelate):
    def __init__(self, input, kernel, stride, mode="VALID"):
        kernel_width = kernel.shape[3] - 1
        kernel_height = kernel.shape[2] - 1
        kernel = jnp.flip(kernel, axis=(2, 3))
        input =  jnp.pad(input, pad_width=((0,0),(0,0),(kernel_height, kernel_height), (kernel_width, kernel_width)), mode='constant')
        
        super().__init__(input, kernel, stride, mode)


class upsampling_gradient:
    def __new__(cls, gradient, stride):
        s_x, s_y = stride
        i, rows, cols = gradient.shape

        dilated_rows = (rows - 1) * s_y + 1
        dilated_cols = (cols - 1) * s_x + 1

        def upsampling(channel_grad):
            dilated = jnp.zeros((dilated_rows, dilated_cols))
            dilated = dilated.at[::s_y, ::s_x].set(channel_grad)
            return dilated

        upsampling_per_kernel = jax.vmap(upsampling, in_axes=0)
        dilate = jax.vmap(upsampling_per_kernel, in_axes=0)

        return dilate(gradient[None, ::, ::, ::]).squeeze(axis=0)

