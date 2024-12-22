#%%

import numpy as np
import jax.numpy as jnp 
from jax import vmap  
from jax import lax 
from jax.lax import dynamic_slice
from jax.scipy.stats import mode as jnp_mode
from jax.numpy import pi as pi
from jax.numpy import e as e

class Kernel():
  def __init__(self, shape: tuple, type: str, **kwargs: dict):
    self.shape = shape
    self.type = type.lower()
    self.array = None
    self.kwargs = kwargs

    # checking for available kernel types only
    types = ["mode","mean", "median", "gaussian", "prewitt", "sobel", "custom"]

    if self.type in types:
      eval(f"self._{self.type}()")

    # raising error
    else:
      raise TypeError(f"The filter of type '{self.type}' is not supported.")

  def _isEven(self):
    if self.shape[0] % 2 == 0 or self.shape[1] % 2 == 0: 
      self.isEven = True
      return True
    else: 
      self.isEven = False
      return False

  # mean kernel
  def _mean(self):
    self.array = jnp.full(self.shape, 1/(self.shape[0]**2))

  # sobel kernel
  def _sobel(self):
    
    # raise error -> kernel != (k,k)
    if self.shape[0] != self.shape[1]:
      raise TypeError(f"For the Sobel kernel, you must pass a square shape, e.g. (3, 3).")

    else:
      if self._isEven():
        self.num = (self.shape[1]) / 2  # while even
        self.index_bound = np.concatenate((np.arange(-self.num, 0, 1), np.arange(0+1, self.num + 1, 1)))

      else:  
        self.num = (self.shape[1] - 1) / 2  # while odd
        self.index_bound = np.concatenate((np.arange(-self.num, 0, 1), np.arange(0, self.num + 1, 1)))

    # kernel(col) -> f(col): where f = colbydistance_Sobel
    def colbydistance_Sobel(self, coluna):
      step = self.shape[1]
      
      # even approach
      if self.isEven == True:
        if coluna == 0: return np.zeros(self.shape[1])

        value = abs(coluna)
        step = (step) / 2
        value = self.num + value

        lower = np.arange(value - step, value + 1, 1)
        upper = np.arange(value - 1, value - step - 1, -1)

        if coluna < 0: return np.concatenate((lower, upper))
        else: return np.concatenate((lower, upper)) * -1

      # odd approach
      else:
        if coluna == 0: return np.zeros(self.shape[1])

        value = abs(coluna)
        step = (step - 1) / 2
        value = self.num + value

        lower = np.arange(value - step, value + 1, 1)
        upper = np.arange(value - 1, value - step - 1, -1)

        if coluna < 0: return np.concatenate((lower, upper))
        else: return np.concatenate((lower, upper)) * -1
            
    # slicing to match the proper shape
    def overallslice_Sobel(self):
      # takin all the results provided by colbydistance
      results = np.array([colbydistance_Sobel(self, index) for index in self.index_bound])
      max_length = results.shape[1]  # max size for cols
      overall_data = np.zeros((len(self.index_bound), max_length))

      # filling the gaps
      for i, res in enumerate(results):
        overall_data[i, :len(res)] = res

      # Sobel kernel for an odd shape e.g., (3x3)
      self.sobel_x = overall_data.T
      self.sobel_y = overall_data

      # correcting the shape and slicing it if it is an even shape.
      if self.isEven == True:
        p1_x = lax.dynamic_slice(self.sobel_x, (0, 0), (int(self.shape[0]/2), self.shape[1]))
        p2_x = lax.dynamic_slice(self.sobel_x, (int(self.shape[0]/2+1), 0), (int(self.shape[0]/2), self.shape[1]))
        self.sobel_x = jnp.concat((p1_x, p2_x))

        #self.sobel_x = lax.dynamic_slice(sobel_x, (0, 0), (self.shape[0]), self.shape[1])

        p1_y = lax.dynamic_slice(self.sobel_y, (0, 0), (self.shape[0], int(self.shape[0]/2)))
        p2_y = lax.dynamic_slice(self.sobel_y, (0, int(self.shape[0]/2+1)), (self.shape[0], int(self.shape[0]/2)))
        self.sobel_y = jnp.concat((p1_y, p2_y), axis=1)

        #self.sobel_y = lax.dynamic_slice(sobel_y, (0, 0), (self.shape[0]), self.shape[1])

    overallslice_Sobel(self) # run the function
    
    # enabling x and y one per time
    if self.kwargs["x"] == True: self.array = self.sobel_x
    elif self.kwargs["y"] == True: self.array = self.sobel_y

  # gaussian kernel
  def _gaussian(self):

    # use only sigma if avaiable
    if "sigma" in self.kwargs.keys():
      self.sigma_x = self.sigma_y = self.kwargs["sigma"]

    else:
      if "sigma_x" in self.kwargs.keys() and "sigma_y" in self.kwargs.keys():
        self.sigma_x = self.kwargs["sigma_x"]
        self.sigma_y = self.kwargs["sigma_y"]
      
      else:
        raise TypeError("You must pass a sigma parameter, or both a sigma_x and a sigma_y value")

    # gaussian function -> G(x,y)
    def G(x, y):
      mu_x = mu_y = (self.shape[0]-1)/2
      k = 1/(2*pi*self.sigma_x*self.sigma_y)
      x_term = (x - mu_x)**2/(self.sigma_x**2)
      y_term = (y - mu_y)**2/(self.sigma_y**2)
      return k*e**(-0.5*(x_term+y_term))

    # gaussian function with vmap along an given vector
    def G_vmap (y):
      return vmap(G, in_axes=(None, 0))(y, jnp.arange(self.shape[1]))

    self.array = vmap(G_vmap)(jnp.arange(self.shape[0]))

  # median kernel
  def _median(self):
    pass

  # custom function for non conventional kernel types
  def _custom(self):
      self.array = jnp.array(self.kwargs["kernel_array"])

  # mode kernel
  def _mode(self):
    pass

  # prewitt kernel
  def _prewitt(self):
    # even kernel -> 2x2, 4x4, ... (2n)x?? or ??x(2n)
    if self._isEven():
      ones_right = int((self.shape[1])/2)

      self.array=jnp.ones((self.shape[0],ones_right))
      self.array = jnp.pad(self.array, pad_width=((0, 0), (0, ones_right)), mode='constant', constant_values=-1) # adding -1 to right side

    # odd kernel -> 3x3, 5x5, ...
    else:
      ones_right = int((self.shape[1] - 1)/2)
      ones_left = int((self.shape[1] - 1)/2)
      self.array = jnp.zeros((self.shape[0],1))
      self.array = jnp.pad(self.array, pad_width=((0, 0), (0, ones_right)), mode='constant', constant_values=-1) # adding -1 to right side
      self.array =jnp.pad(self.array, pad_width=((0, 0), (ones_left, 0)), mode='constant', constant_values=1)    # adding 1 to left side

  

class Morphology():
  def __init__(self, image:np.ndarray, kernel: tuple, operation:str):
    self.image = jnp.array(image)
    self.operations = ["dilation", "erosion", "opening", "closing"]
    self.kernel = jnp.ones((kernel))

    processed = eval(f"self.{operation}(array=self.image)")
    self.result = self._setBinary(array = processed)

  def dilation(self, array:jnp.ndarray):
    return Filter2D(image=array, filter="custom", kernel_array=self.kernel, operation="morph_dilation").filtered

  def erosion(self, array:jnp.ndarray):
    return Filter2D(image=array, filter="custom", kernel_array=self.kernel, operation="morph_erosion").filtered

  def opening(self, array:jnp.ndarray):
    return self.dilation(array=self.erosion(array=array))

  def closing(self,array:jnp.ndarray):
    return self.erosion(array=self.dilation(array=array))

  def _setBinary(self, array):
    return jnp.where(array==True, 255, jnp.where(array==False, 0, array))
  


# classical filtering function
class Filter2D():
  def __init__(self, image:np.ndarray, filter:str, kernel:tuple=None, **kwargs:dict):
    self.filter = filter
    self.kwargs = kwargs
    self.image = jnp.array(image)
    self.A_star = self.image
    self.kernel = kernel

    self.m, self.n = self.image.shape   # (m,n) - image shape

    # properly approach for i,j value for all the possible inputs
    self._ijCorrection()
    

    # starting the functions
    self._padding()
    C_m = jnp.arange(self.upper_m, self.upper_m + self.m) # the coordinates of rows' image [Cm]

    # checking if the kernel type is equal to sobel
    if filter == "sobel":

      # checking for valid threshold input
      if "threshold" in self.kwargs.keys() and (type(self.kwargs["threshold"]) == int or type(self.kwargs["threshold"]) == float):
        self.threshold = self.kwargs["threshold"]

      else:
        raise TypeError(f"Error: expected a threshold parameter and an integer value")


      self.kernel = Kernel(shape = kernel, type=filter, x=True, y=False, **kwargs)
      self.filtered_sobelX = vmap(self._stackRow)(C_m)

      self.kernel = Kernel(shape = kernel, type=filter, x=False, y=True, **kwargs)
      self.filtered_sobelY = vmap(self._stackRow)(C_m)

      self.filtered = self.gradient2d(X_1=self.filtered_sobelX, X_2=self.filtered_sobelY)
      self.filtered = self.binary_threshold(array=self.filtered, threshold=self.threshold)

    # normal kernel
    else:
      self.kernel = Kernel(shape = kernel, type=filter, **kwargs)
      self.filtered = vmap(self._stackRow)(C_m)

  # sum function applied to the sliding window and kernel multiplication.
  # return an encapsulated scalar value in JAX format
 
  def _singleFilter(self, window):

    if self.filter == "custom" and "operation" in self.kwargs.keys():
      if self.kwargs["operation"] == "morph_dilation": return jnp.any(window) # dilation
      elif self.kwargs["operation"] == "morph_erosion": return jnp.all(window) # erosion

    elif self.kernel.type == "median": return jnp.median(window) # median applied

    elif self.kernel.type == "mode":
      mode, count = jnp_mode(jnp.ravel(window))
      return mode

    else: return jnp.sum(window * self.kernel.array) # convolution applied

  # ij correction function to handle all kernel types
  def _ijCorrection(self):
    # custom filter
    if self.filter == "custom":
      if "kernel_array" in self.kwargs.keys():
        self.kernel = jnp.array(self.kwargs["kernel_array"]).shape
        self.i, self.j = self.kernel 

      # raising error
      else:
        raise TypeError(f"You must pass and kernel_array parameter")

    # non custom filters
    else:
      # raising error
      if type(self.kernel) != tuple:
        raise TypeError(f"You must pass a valid tuple argument for the kernel parameter")
      else:
        self.i, self.j = self.kernel  # (i,j) - kernel shape

  def _w(self, x, y): 
    # x = row and y = col    

    # even kernel
    if self.i % 2 == 0 or self.j % 2 == 0: 
      return lax.dynamic_slice(self.A_star, (x, y), (self.i, self.j)) # returning window for an even kernel

    # odd kernel
    else: 
      return lax.dynamic_slice(self.A_star, (x-1, y-1), (self.i, self.j)) # returning window for an odd kernel
        
    
  def _delta(self, x, y):
     # x = row and y = col    
    window = self._w(x, y)
     # taking the window with the size of the kernel based on (x,y) pos.
    return self._singleFilter(window)

    # At the end of the iteration, the delta() function will produce an 1D array of shape n, which corresponds to the result of the convolution applied to the elements of the row using a sliding window at each (row, col) position.

  def _stackRow(self, x):
    # x = row 
    C_n = jnp.arange(self.upper_n, self.upper_n + self.n) # the coordinates of columns'image  [Cn]
    return vmap(self._delta, in_axes=(None, 0))(x, C_n) # row -> set as fixed during the vmap

    # at the final the _stackrow, will stack all the results provided by _delta()

  def _padding(self):
    self.is_even = False
    # even kernel
    if self.i % 2 == 0 or self.j % 2 == 0:
      self.upper_m = int(self.i - 1) # number of rows above my image
      self.upper_n = int(self.j - 1) # number of cols above my image

      self.A_star = jnp.pad(self.A_star, pad_width=((1, self.upper_m), (1, self.upper_n)), mode='edge')
      self.upper_m = self.upper_n = 1 # to solve the C_n and C_m issue for even kernels

    # odd kernel


    else:
      self.upper_m = int(( self.i - 1)/2) # number of rows above my image
      self.upper_n = int(( self.j - 1)/2) # number of cols above my image
      self.A_star = jnp.pad(self.A_star, pad_width=((self.upper_m, self.upper_m), (self.upper_n, self.upper_n)), mode='edge')
        
  def gradient2d(self, X_1, X_2):
    # X_1: first image array // X_2: second image array
    return jnp.sqrt(jnp.square(X_1) + jnp.square(X_2))

  def binary_threshold(self, threshold, array):
    return jnp.where(array>=threshold, 255, jnp.where(array<threshold, 0, array))

