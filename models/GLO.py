import numpy as np
from copy import deepcopy
from chainer import optimizers, Variable
import matplotlib.pyplot as plt

from models.networks import BasicGLOGeneratorNetwork

class GLOModel():
    def __init__(self, code_len, n_vecs, minibatch_size, dataset, n_pixels, g_hidden, z_learning_rate=0.01,
                 g_learning_rate=0.05):
        g_input_size = code_len
        g_hidden_size = g_hidden
        g_output_size = n_pixels

        z_output_size = code_len
        self.minibatch_size = minibatch_size
        self.dataset = dataset

        # representation space gaussian distr to be learnt from learnable latent vectors Z
        self.rspace_mean = []
        self.rspace_std = []

        # like GAN, GLO seeks to optimize how to generate a reconstruction of data
        # from input seed vectors by optimizing a generator G.
        # unlike GAN, GLO tries to optimize these initial seed-vectors as well!
        # Z are these latent vectors

        self.Z = initialZ(n_vecs, z_output_size, dataset, method='pca')
        self.G = BasicGLOGeneratorNetwork(g_hidden_size, g_output_size, n_input=z_output_size)

        self.g_optimizer = optimizers.SGD(lr=g_learning_rate)
        self.g_optimizer.setup(self.G)
        self.z_optimizer = optimizers.SGD(lr=z_learning_rate)
        self.z_optimizer.setup(self.G.z)


    def train(self, lossfun, n_epochs=100):
        print('start training loop')

        for epoch in range(0, n_epochs):

            losses = []

            self.g_optimizer.new_epoch()
            self.z_optimizer.new_epoch()
            for i in range(0, len(self.dataset)):
                data = self.dataset[i][0]

                self.G.cleargrads()

                recon = self.G(self.Z[i])
                loss = lossfun(recon, np.atleast_2d(data))  # loss is discriminator-estimated 'fakeness'
                loss.backward(retain_grad=True)
                self.g_optimizer.update()
                self.z_optimizer.update()

                self.Z[i,:] = project_z_to_ball(self.G.getZ() - 0.1*self.G.z.z.grad)

                losses.append(loss)
        print('done!')

        # calculate the gaussian distr of the learnt representation space
        # this way, we can pull random samples from it
        # that will work with the network
        print(np.mean(self.Z, axis=0))
        print(np.std(self.Z, axis=0))
        self.rspace_mean = np.mean(self.Z, axis=0)
        self.rspace_std = np.std(self.Z, axis=0)
        print(len(self.rspace_mean), len(self.rspace_std))


    def generate_unseen(self, n):
        seeds = []
        unseen = []
        for ni in range(0, n):
            x = [np.random.normal(self.rspace_mean[i], self.rspace_std[i]) for i in range(0, len(self.rspace_mean))]
            x = np.array(x, dtype='float32')
            seeds.append(x)
            unseen.append(self.G(x))

        return seeds, unseen

def initialZ(n_vecs, z_output_size, dataset, method='random'):
    if method == 'random':
        return np.random.randn(n_vecs, z_output_size).astype('float32')
    elif method == 'pca':
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        data = np.array([d for (d, l) in dataset])
        data = StandardScaler().fit_transform(data.T)

        pca = PCA(n_components=z_output_size)
        z = pca.fit_transform(data)
        return z
    else:
        return np.random.randn(n_vecs, z_output_size).astype('float32')


def project_z_to_ball(z):
    m = np.maximum(np.sqrt(np.sum(z**2)), 1)
    return z / m


def showprepostz(prez, postz):
    sqrt_pixels = int(np.sqrt(len(prez)))
    print(len(prez), sqrt_pixels)
    diffz = postz - prez

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(prez.reshape((sqrt_pixels, sqrt_pixels)))
    axarr[0].set_title('latent vector z pre-update')
    axarr[1].imshow(postz.reshape((sqrt_pixels, sqrt_pixels)))
    axarr[1].set_title('latent vector z post-update')
    axarr[2].imshow(diffz.reshape((sqrt_pixels, sqrt_pixels)))
    axarr[2].set_title('difference pre and post-update')
    plt.show()
