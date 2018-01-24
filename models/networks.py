from  chainer import Chain
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
