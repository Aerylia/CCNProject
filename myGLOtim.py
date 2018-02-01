
#%%

import numpy as np  
import random
import copy

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.preprocessing import normalize

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, initializers
from chainer import Link, Chain, ChainList, Parameter
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import TupleDataset

#%%

def get_mnist(n_train=100, n_test=100, n_dim=1, with_label=True, classes = None):
    """

    :param n_train: nr of training examples per class
    :param n_test: nr of test examples per class
    :param n_dim: 1 or 3 (for convolutional input)
    :param with_label: whether or not to also provide labels
    :param classes: if not None, then it selects only those classes, e.g. [0, 1]
    :return:
    """

    train_data, test_data = chainer.datasets.get_mnist(ndim=n_dim, withlabel=with_label, dtype='float32')
    train_data = train_data
    test_data = test_data

    if not classes:
        classes = np.arange(10)
    n_classes = len(classes)

    if with_label:

        for d in range(2):

            if d==0:
                data = train_data._datasets[0]
                labels = train_data._datasets[1]
                n = n_train
            else:
                data = test_data._datasets[0]
                labels = test_data._datasets[1]
                n = n_test

            for i in range(n_classes):
                lidx = np.where(labels == classes[i])[0][:n]
                if i==0:
                    idx = lidx
                else:
                    idx = np.hstack([idx,lidx])

            L = np.concatenate([i*np.ones(n) for i in np.arange(n_classes)]).astype('int32')

            if d==0:
                train_data = TupleDataset(data[idx],L)
            else:
                test_data = TupleDataset(data[idx],L)

    else:

        tmp1, tmp2 = chainer.datasets.get_mnist(ndim=n_dim,withlabel=True)

        for d in range(2):

            if d == 0:
                data = train_data
                labels = tmp1._datasets[1]
                n = n_train
            else:
                data = test_data
                labels = tmp2._datasets[1]
                n = n_test

            for i in range(n_classes):
                lidx = np.where(labels == classes[i])[0][:n]
                if i == 0:
                    idx = lidx
                else:
                    idx = np.hstack([idx, lidx])

            if d == 0:
                train_data = data[idx]
            else:
                test_data = data[idx]

    return train_data, test_data
    
#%%

class BasicGeneratorNetwork(Chain):
    def __init__(self, n_hidden, n_out, n_input=None):
        super(BasicGeneratorNetwork, self).__init__()
        self.n_input = n_input
        with self.init_scope():
            self.z = L.Parameter(np.random.randn(1, self.n_input))
            self.l1 = L.Linear(n_input, n_input)
            self.l2 = L.BatchNormalization(n_input)
            self.l3 = L.Linear(n_input, n_out)

    def __call__(self, x):
        #print(zinput)
        self.l1.b.data = x.data.flatten()
        
        s = np.ones((1, self.n_input), dtype='float32')
        h1 = F.relu(self.l1(s))
        h2 = self.l2(h1)
        y = F.sigmoid(self.l3(h2))
        return y   
        
#%%

def showprepostz(prez, postz):
    
    prez = prez.flatten()
    postz = postz.flatten()
    sqrt_pixels = np.sqrt(len(prez))
    print(len(prez), sqrt_pixels)
    diffz = postz - prez
    
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(prez.reshape((sqrt_pixels,sqrt_pixels)))
    axarr[0].set_title('latent vector z pre-update')
    axarr[1].imshow(postz.reshape((sqrt_pixels,sqrt_pixels)))
    axarr[1].set_title('latent vector z post-update')
    axarr[2].imshow(diffz.reshape((sqrt_pixels,sqrt_pixels)))
    axarr[2].set_title('difference pre and post-update')
    plt.show()
        
#%%

def initialZ(n_vecs, z_output_size, dataset, method = 'random'):
    if method == 'random':
        return np.random.randn(n_vecs, z_output_size).astype('float32')
    elif method == 'pca':
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        data = np.array([d for (d, l) in dataset])
        data = StandardScaler().fit_transform(data)
        
        pca = PCA(n_components=z_output_size)
        z = pca.fit_transform(data)
        
        return z
    else:
        return np.random.randn(n_vecs, z_output_size).astype('float32')
        
def project_z_to_ball(z):
    m = np.maximum(np.sqrt(np.sum(z**2)), 1)
    return z / m

class GLOModel():

    def __init__(self, code_len, n_vecs, minibatch_size, dataset, n_pixels, g_hidden, z_learning_rate=0.1, g_learning_rate=1, init = 'random'):
        
        g_input_size = code_len
        g_hidden_size = 10
        g_output_size = n_pixels

        self.n_vecs = n_vecs
        self.z_output_size = code_len
        self.z_learning_rate = z_learning_rate
        self.minibatch_size = minibatch_size
        self.dataset = dataset
        self.init = init
        
        # representation space gaussian distr to be learnt from learnable latent vectors Z
        self.rspace_mean = []
        self.rspace_std = []
        self.losses = []
        
        # like GAN, GLO seeks to optimize how to generate a reconstruction of data
        # from input seed vectors by optimizing a generator G.
        # unlike GAN, GLO tries to optimize these initial seed-vectors as well!
        # Z are these latent vectors

        self.Z = initialZ(n_vecs, code_len, dataset, method=init)
        
        self.G = BasicGeneratorNetwork(g_hidden_size, g_output_size, n_input=code_len)

        self.g_optimizer = optimizers.SGD(lr=g_learning_rate)
        self.g_optimizer.setup(self.G)

    def train(self, lossfun, n_epochs = 100):

        print('start training loop')

        for epoch in range(0, n_epochs):
            
            losses = []

            self.g_optimizer.new_epoch()

            for i in range(0, self.n_vecs):
                
                data = self.dataset[i][0]
                zi = Variable(np.atleast_2d(self.Z[i]), requires_grad = True)
                
                self.G.cleargrads()
                
                recon = self.G(zi)
                loss = lossfun(recon, np.atleast_2d(data)) 
                loss.backward(retain_grad = True)
                
                prez = np.copy(zi.data)
                
                if (epoch % int(n_epochs/10) == 0) and i == 0:
                    print('before')
                    print(self.G.l1.b.data)
                
                self.g_optimizer.update()
                
                if (epoch % int(n_epochs/10) == 0) and i == 0:
                    print('after')
                    print(self.G.l1.b.data)
                
                self.Z[i] = project_z_to_ball(self.G.l1.b.data)
                postz = np.copy(self.Z[i])
                
                losses.append(loss)
                
            self.losses.append((sum(losses)/len(losses)).data)
                
            if epoch % int(n_epochs/10) == 0:
                print('epoch: {0}/{1}: loss: {2}'.format(epoch,n_epochs, sum(losses)/len(losses)))
                showprepostz(prez, postz)

        print('done!')
        
        # calculate the gaussian distr of the learnt representation space
        # this way, we can pull random samples from it
        # that will work with the network
        self.rspace_mean = np.mean(self.Z, axis=0)
        self.rspace_std = np.std(self.Z, axis=0)
        
    def generate_unseen(self, n):
        seeds = []
        unseen = []
        for ni in range(0, n):
            x = [np.random.normal(self.rspace_mean[i], self.rspace_std[i]) for i in range(0, len(self.rspace_mean))]
            x = np.array(x, dtype='float32')
            seeds.append(x)
            x = Variable(np.atleast_2d(x))
            unseen.append(self.G(x))
            
        return seeds, unseen
        
    def project_to_rspace(self, x):
        x = np.linalg.norm(x)
        x = self.rspace_mean + np.multiply(x, self.rspace_std)
        return x
        
    def test(self, x):
        rspace_x = self.project_to_rspace(x)
        orig = []
        seeds = []
        unseen = []        
        for xi in range(0, len(x)):
            g = self.G(rspace_x[xi])
            orig.append(x[xi])
            seeds.append(rspace_x[xi])
            unseen.append(g)
        return orig, seeds, unseen
        

#%%

def show_results(model):
    
    print('show results')
    
    sqrtz = np.sqrt(len(model.Z[0]))
    sqrt_pixels = np.sqrt(len(model.dataset[0][0]))
    
    for xi in range(0, len(model.dataset), int(len(model.dataset)/10)):
        
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(model.Z[xi].reshape((sqrtz,sqrtz)))
        axarr[0].set_title('latent vector z {0}'.format(xi))
        axarr[1].imshow(model.dataset[xi][0].reshape((sqrt_pixels,sqrt_pixels)))
        axarr[1].set_title('real data vector x {0}'.format(xi))
        axarr[2].imshow(model.G(Variable(np.atleast_2d(model.Z[xi]))).data.reshape((sqrt_pixels,sqrt_pixels)))
        axarr[2].set_title('reconstructed data vector g {0}'.format(xi))
        plt.show()
    
def show_unseen(model, n):
    
    print('show unseen')
    
    seeds, unseen = model.generate_unseen(n)
    
    sqrtz = np.sqrt(len(model.Z[0]))
    sqrt_pixels = int(np.sqrt(len(model.dataset[0][0])))
    
    for xi in range(0, n):
        
        us = np.array(unseen[xi][0].data)
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(seeds[xi].reshape((sqrtz,sqrtz)))
        axarr[0].set_title('rspace vector seed {0}'.format(xi))
        axarr[1].imshow(us.reshape((sqrt_pixels,sqrt_pixels)))
        axarr[1].set_title('generated vector {0}'.format(xi))
        plt.show()
        
#%%

train, test = get_mnist(classes = [5])
n_vectors = len(train)
n_pixels = len(train[0][0])
code_dim = 64
lossfun = F.mean_squared_error
init = 'pca'
model = GLOModel(code_dim, n_vectors, 1, train, n_pixels, 50, init = init)
model.train(lossfun)
    
show_results(model)

#%%

print(model.rspace_mean.shape)
print(model.rspace_std.shape)

show_unseen(model, 10)

#%%

print(model.losses)
plt.plot(range(0, len(model.losses)), model.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('loss vs epoch in GLO')

#%%

class CycleGLO():

    def __init__(self, alpha, beta, lambda1, lambda2, n_pixels, learning_rate_decay, learning_rate_interval, g_hidden, max_buffer_size, lossfun, data1, data2, codelen1, codelen2, dataset1, dataset2, init = 'random'):
        self.max_buffer_size = max_buffer_size

        g_input_size = n_pixels
        g_hidden_size = g_hidden
        g_output_size = n_pixels
        
        self.codelen1 = codelen1
        self.codelen2 = codelen2
        
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
        self.Z1 = initialZ(n_vecs, codelen1, dataset1, method=init)
        self.Z2 = initialZ(n_vecs, codelen2, dataset2, method=init)

        self.G = GenerativeNetwork(g_hidden_size, g_output_size, n_input=self.codelen1)
        self.F = GenerativeNetwork(g_hidden_size, g_output_size, n_input=self.codelen2)

        self.opt_g = optimizers.Adam(alpha=alpha, beta1=beta)
        self.opt_f = optimizers.Adam(alpha=alpha, beta1=beta)

        self.opt_g.setup(self.G)
        self.opt_f.setup(self.F)

        self.opt_g.use_cleargrads()
        self.opt_f.use_cleargrads()

        self.n_pixels = n_pixels
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_interval = learning_rate_interval

        self.buffer = {}
        self.buffer['x'] = np.zeros((self.max_buffer_size, self.n_pixels)).astype('float32')
        self.buffer['y'] = np.zeros((self.max_buffer_size, self.n_pixels)).astype('float32')
        
        self.lossfun = lossfun

    def getAndUpdateBuffer(self, buffer, data, epoch):
        if epoch < self.max_buffer_size:
            self.buffer[buffer][epoch, :] = data[0]
            return data
        self.buffer[buffer][:-2, :] = self.buffer[buffer][1:-1, :]
        self.buffer[buffer][self.max_buffer_size-1, :] = data[0]

        if np.random.rand() < 0.5:
            return data
        id = np.random.randint(0, self.max_buffer_size)
        return self.buffer[buffer][id, :].reshape((1,self.n_pixels))

    def cycle_loss(self, x, y):
        return F.mean_absolute_error(x, y)

    def train(self, n_epochs, batch_iter):
        print('Start training CycleGLO')
        for epoch in range(n_epochs):
            print(epoch)
            batch = batch_iter.next() # chainer.iterators.MultiProcessIterator
            batchsize = len(batch)
            w_in = self.n_pixels
            x = np.zeros((batchsize, w_in)).astype('float32')
            y = np.zeros((batchsize, w_in)).astype('float32')
            for i in range(batchsize):
                x[i, :] = np.asarray(batch[i][0])
                y[i, :] = np.asarray(batch[i][1])
            x, y = Variable(x), Variable(y)
            #print(x.shape, y.shape)
            #print('Training g')
            xy = self.g(x)
            #print(xy.shape)
            xy_copy = Variable(self.getAndUpdateBuffer('x', xy.data, epoch))
            #print(xy_copy.shape)
            xyx = self.f(xy)
            #print(xyx.shape)

            #print('Training f')
            yx = self.f(y)
            yx_copy = Variable(self.getAndUpdateBuffer('y', yx.data, epoch))
            yxy = self.g(yx)

            if self.learning_rate_decay > 0 and epoch % self.learning_rate_interval == 0 :
                if self.opt_g.alpha > self.learning_rate_decay:
                    self.opt_g.alpha -= self.learning_rate_decay
                if self.opt_f.alpha > self.learning_rate_decay:
                    self.opt_f.alpha -= self.learning_rate_decay
                if self.opt_x.alpha > self.learning_rate_decay:
                    self.opt_x.alpha -= self.learning_rate_decay
                if self.opt_y.alpha > self.learning_rate_decay:
                    self.opt_y.alpha -= self.learning_rate_decay

            self.f.cleargrads()
            self.g.cleargrads()
            self.x.cleargrads()
            self.y.cleargrads()

            #print('Calculating loss')
            #### Calculate the loss:
            # y loss
            y_fake_loss = self.dis_fake_loss(self.y(xy_copy))
            y_real_loss = self.dis_real_loss(self.y(y))
            y_loss = y_fake_loss + y_real_loss

            x_fake_loss = self.dis_fake_loss(self.x(yx_copy))
            x_real_loss = self.dis_real_loss(self.x(x))
            x_loss = x_fake_loss + x_real_loss

            g_loss = self.gen_loss(self.y(xy))
            f_loss = self.gen_loss(self.x(yx))

            cycle_x_loss = self.lambda1 * self.cycle_loss(xyx, x)
            cycle_y_loss = self.lambda1 * self.cycle_loss(yxy, y)
            gen_loss = self.lambda2 * g_loss + self.lambda2 * f_loss + cycle_x_loss + cycle_y_loss
            #### Update
            y_loss.backward()
            x_loss.backward()
            self.opt_x.update()
            self.opt_y.update()

            gen_loss.backward()
            self.opt_g.update()
            self.opt_f.update()

            # TODO report losses

            # TODO store intermediate result every so often.
            if epoch % 5000 == 0:
                print(x_loss, y_loss)
                print(gen_loss, cycle_y_loss, cycle_x_loss)
                sqrt_pixels = int(np.sqrt(self.n_pixels))
                input, generated = y.data, yx
                print(generated.data)
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(input.reshape((sqrt_pixels, sqrt_pixels)))
                axarr[0].set_title('noise input G')
                axarr[1].imshow(generated.data.reshape((sqrt_pixels, sqrt_pixels)))
                axarr[1].set_title('generated sample')
                plt.show()

                input, generated = x.data, xy
                print(generated.data)
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(input.reshape((sqrt_pixels, sqrt_pixels)))
                axarr[0].set_title('noise input F')
                axarr[1].imshow(generated.data.reshape((sqrt_pixels, sqrt_pixels)))
                axarr[1].set_title('generated sample')
                plt.show()

