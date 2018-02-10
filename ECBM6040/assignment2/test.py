# Import modules
from __future__ import print_function
import tensorflow as tf
import numpy as np
from ecbm4040.cifar_utils import load_data
from matplotlib import pyplot as plt

from ecbm4040.layer_funcs import conv2d_forward

# Set test parameters.
x_shape = (2, 4, 4, 3)
w_shape = (3, 4, 4, 3)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)
pad = 1
stride = 2
your_feedforward = conv2d_forward(x, w, b, pad, stride)
print(your_feedforward)
