from .CycleGAN import CycleGAN
from .CycleGLO import CycleGLO
from .GAN import GANModel
from .GLO import GLOModel
import numpy as np


class NoiseSampler():
    def __init__(self, fun=np.random.rand, batch_size=1, *funargs, **funkwargs):
        self.fun = fun
        self.args = funargs
        self.kwargs = funkwargs

    def __call__(self, batch_size):
        return np.array([self.fun(*self.args, **self.kwargs).astype('float32') for i in range(0, batch_size)])