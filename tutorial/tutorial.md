
# Tutorial - Filtering 
The default filters available in the current version of CCV are:

| Smoothing          | Edge Detection        |
|-------------------|-------------------|
|  mean         | prewitt        |
|  median        | sobel         |
|  mode        |          |
|  gaussian        |          |

## Warning
To run any of the codes presented in this tutorial without issues in your Jupyter Notebook, you must clone this repository using the following command:
```python
!git clone https://github.com/ClaytonSdS/ccv.git
```

## Smoothing Filters
If our objective is to achieve a blur image or reduce noise, we can use some of the available smoothing kernels, such as "mean" or "gaussian".  
For example, let's implement a simple filtering process using a mean kernel with a 5x5 shape:
  
```python
import ccv
from skimage import io

# Reading your image and converting it to grayscale.
image_path = 'ccv/tutorial/car.png'  # your image path
img = io.imread(image_path, as_gray=True)

# Applying the mean filter.
image_filter = ccv.Filter2D(image=img, filter="mean", kernel=(5,5))
result = image_filter.filtered

```
During this process, the kernel that we are using to convolve with the img variable is the same as the one in the below equation.

  
```math
Kernel = \frac{1}{25} \begin{bmatrix}
1&1&1&1&1\\
1&1&1&1&1\\
1&1&1&1&1\\
1&1&1&1&1\\
1&1&1&1&1\\
\end{bmatrix}
```

For checking the results, you can compare both images using the Matplotlib library, as well as:
```python
import matplotlib.pyplot as plt 

plt.figure(figsize=(30, 20))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```
#### Figure 1: Filtering image with mean kernel.
<img src="https://github.com/ClaytonSdS/ccv/blob/main/tutorial/mean_filter.jpg"/>

Moreover, you can use more complex filters like gaussian, which accepts both (σ<sub>x</sub>, σ<sub>y</sub>) or just σ as parameter.  
One possible example using both σ<sub>x</sub> and σ<sub>y</sub> is: 
  
```python
import ccv
import jax.numpy as jnp

# Setting a random image with jax.numpy.ones
img = jnp.ones((1024,768))

# Applying the Gaussian filter with different sigmas
gaussian_xy = ccv.Filter2D(image=img, filter="gaussian", kernel=(3,3), sigma_x=20, sigma_y=30)
result = gaussian_xy.filtered

```
A larger kernel which represents a Gaussian with σ<sub>x</sub> = 20 and σ<sub>y</sub> = 30 is shown in Figure 2.
#### Figure 2: Gaussian kernel with different sigmas.
<img src="https://github.com/ClaytonSdS/ccv/blob/main/tutorial/gaussian_xy.jpg" width="500"/>

On the other hand, we can use only a σ value if our goal is to establish a circular shape centered on the kernel's center.
<div style="border: 1px solid black; padding: 10px;">
  
```python
import ccv
import jax.numpy as jnp

# Setting a random image with jax.numpy.ones
img = jnp.ones((1024,768))

# Applying the Gaussian filter with equal sigmas
gaussian_sigma = ccv.Filter2D(image=img, filter="gaussian", kernel=(3,3), sigma=30)
result = gaussian_sigma.filtered

```
This example of a circular kernel can be visualized in Figure 3, which represents a larger kernel (100, 100) but with an equal sigma set to 30.
#### Figure 3: Gaussian kernel with equal sigmas.
<img src="https://github.com/ClaytonSdS/ccv/blob/main/tutorial/gaussian_sigma.jpg" width="500"/>

## Edge Detection Filters
In the current version, the only two built-in implementations of edge detection filters are Sobel and Prewitt.  
Thus, dealing with Sobel filter one possible implementation can be:

```python
from skimage import io
import ccv

# Reading your image
image_path = 'ccv/tutorial/car.png' # your image path
img = io.imread(image_path, as_gray=True)

# Applying the Sobel filter with a threshold equal to 0.8
sobel_image = ccv.Filter2D(image=img, filter="sobel", kernel=(3,3), threshold = 0.8)
result = sobel_image.filtered

```
Notice that the ccv.Filter2D method convolves the image in both the x and y directions, calculating the gradient and its magnitude, and the final result is an array obtained by applying a threshold function to the magnitude of the gradient at each (x, y) coordinate, which produces a black-and-white image as we can see on figure 4.   

#### Figure 4: Filtering image with sobel kernel.
<img src="https://github.com/ClaytonSdS/ccv/blob/main/tutorial/sobel_filter.jpg"/>

## Custom Filters
If you want to use your custom kernel, you must pass it using the "kernel_array" parameter and set the filter parameter as "custom".   
For example, you can specify your custom filter as:

```python
from skimage import io
import ccv

# Reading your image
image_path = 'ccv/tutorial/car.png' # your image path
img = io.imread(image_path, as_gray=True)

# Applying your custom filter
custom_filter = jnp.full((10,10), 1/(10**2))
custom_image = ccv.Filter2D(image=img, filter="custom", kernel_array=custom_filter)
result = np.array(custom_image.filtered)


```
