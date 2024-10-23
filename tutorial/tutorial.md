
# Tutorial - Filtering 
The default filters available in the current version of CCV are:

| Smoothing          | Edge Detection        |
|-------------------|-------------------|
|  mean         | prewitt        |
|  median        | sobel         |
|  mode        |          |
|  gaussian        |          |

## Smoothing Filters
If our objective is to achieve a blur image or reduce noise, we can use some of the available smoothing kernels, such as "mean" or "gaussian".  
For example, let's implement a simple filtering process using a mean kernel with a 3x3 shape:
<div style="border: 1px solid black; padding: 10px;">
  
```python
import ccv
import jax.numpy as jnp
  
img = jnp.ones((1024,768))
image_filter = ccv.Filter2D(image=img, filter="mean", kernel=(3,3))
result = image_filter.filtered

```
During this process, the kernel that we are using to convolve with the img variable is the same as the one in the below equation.


<div style="text-align: center;">
  
```math
Kernel = \frac{1}{9}\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1\\
\end{bmatrix}
```

Notice that the result, it's a jax array type and for plotting using matplotlib you might need to convert it to a numpy array like.  


Moreover, you can use more complex filters like gaussian, which accepts both (σ<sub>x</sub>, σ<sub>y</sub>) or just σ as parameter.  
As an example using both σ<sub>x</sub> and σ<sub>y</sub>: 
<div style="border: 1px solid black; padding: 10px;">
  
```python
import ccv
import jax.numpy as jnp
  
img = jnp.ones((1024,768))
gaussian_xy = ccv.Filter2D(image=img, filter="gaussian", kernel=(3,3), sigma_x=20, sigma_y=30)
result = gaussian_xy.filtered

```
The larger kernel which represents a Gaussian with sigma_x = 20 and sigma_y = 30 is shown in Figure 1..
#### Figure 1: Gaussian kernel with different sigmas.
<img src="https://github.com/ClaytonSdS/ccv/blob/main/tutorial/gaussian_xy.jpg" width="500"/>

On the other hand, we can use only a σ value if our goal is to establish a circular shape centered on the kernel's center.
<div style="border: 1px solid black; padding: 10px;">
  
```python
import ccv
import jax.numpy as jnp
  
img = jnp.ones((1024,768))
gaussian_sigma = ccv.Filter2D(image=img, filter="gaussian", kernel=(3,3), sigma=30)
result = gaussian_sigma.filtered

```
This example of a circular kernel can be visualized in Figure 2, which represents a larger kernel (100, 100) but with an equal sigma set to 30.
#### Figure 2: Gaussian kernel with equal sigmas.
<img src="https://github.com/ClaytonSdS/ccv/blob/main/tutorial/gaussian_sigma.jpg" width="500"/>

## Edge Detection Filters


## Custom Filters
