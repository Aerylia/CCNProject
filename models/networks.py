import numpy as np
from  chainer import Chain, Link, initializers, Parameter
import chainer.functions as F
import chainer.links as L

from PIL import Image

class BasicDiscriminatorNetwork(Chain):
    def __init__(self, n_hidden, n_out, n_input=None):
        super(BasicDiscriminatorNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_input, n_hidden)
            self.l2 = L.Linear(n_hidden, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y


class BasicGeneratorNetwork(Chain):
    def __init__(self, n_hidden, n_out, n_input=None):
        super(BasicGeneratorNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_input, n_hidden)
            self.l2 = L.BatchNormalization(n_hidden)
            self.l3 = L.Linear(n_hidden, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = self.l2(h1)
        y = F.sigmoid(self.l3(h2))
        return y


class IDLink(Link):
    def __init__(self, n_out):
        super(IDLink, self).__init__()
        with self.init_scope():
            self.z = Parameter(initializer=initializers.Uniform(), shape=(n_out, 1))

    def setZ(self, x):
        self.z.data = x

    def __call__(self):
        return self.z.data

class BasicGLOGeneratorNetwork(Chain):
    def __init__(self, n_hidden, n_out, n_input=None):
        super(BasicGLOGeneratorNetwork, self).__init__()
        with self.init_scope():
            self.z = IDLink(n_input)
            self.l1 = L.Linear(n_input, n_hidden)
            self.l2 = L.BatchNormalization(n_hidden)
            self.l3 = L.Linear(n_hidden, n_out)

    def __call__(self, x):
        self.setZ(x)
        h1 = F.relu(self.l1(np.atleast_2d(self.z())))
        h2 = self.l2(h1)
        y = F.sigmoid(self.l3(h2))
        return y

    def setZ(self, x):
        return self.z.setZ(x)

    def getZ(self):
        return self.z.z.data

    def cleargrads(self):
        super(BasicGLOGeneratorNetwork, self).cleargrads()
        self.l1.cleargrads()
        self.l2.cleargrads()
        self.l3.cleargrads()