from chainer import optimizers, Variable
import chainer.functions as F
import numpy as np
import matplotlib.pyplot as plt

from models.networks import BasicGeneratorNetwork as GenerativeNetwork
from models.networks import BasicDiscriminatorNetwork as Discriminator

class CycleGAN():

    def __init__(self, alpha, beta, lambda1, lambda2, n_pixels, learning_rate_decay, learning_rate_interval, g_hidden, d_hidden, max_buffer_size):
        self.max_buffer_size = max_buffer_size

        g_input_size = n_pixels
        g_hidden_size = g_hidden
        g_output_size = n_pixels

        d_input_size = g_output_size
        d_hidden_size = d_hidden
        d_output_size = 1

        self.g = GenerativeNetwork(g_hidden_size, g_output_size, n_input=g_input_size)
        self.f = GenerativeNetwork(g_hidden_size, g_output_size, n_input=g_input_size)
        self.x = Discriminator(d_hidden_size, d_output_size, n_input=d_input_size)
        self.y = Discriminator(d_hidden_size, d_output_size, n_input=d_input_size)

        self.opt_g = optimizers.Adam(alpha=alpha, beta1=beta)
        self.opt_f = optimizers.Adam(alpha=alpha, beta1=beta)
        self.opt_x = optimizers.Adam(alpha=alpha, beta1=beta)
        self.opt_y = optimizers.Adam(alpha=alpha, beta1=beta)
        #self.opt_g = optimizers.SGD(alpha)
        #self.opt_f = optimizers.SGD(alpha)
        #self.opt_x = optimizers.SGD(alpha)
        #self.opt_y = optimizers.SGD(alpha)

        self.opt_g.setup(self.g)
        self.opt_f.setup(self.f)
        self.opt_x.setup(self.x)
        self.opt_y.setup(self.y)

        self.opt_g.use_cleargrads()
        self.opt_f.use_cleargrads()
        self.opt_x.use_cleargrads()
        self.opt_y.use_cleargrads()

        self.n_pixels = n_pixels
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_interval = learning_rate_interval

        self.buffer = {}
        self.buffer['x'] = np.zeros((self.max_buffer_size, self.n_pixels)).astype('float32')
        self.buffer['y'] = np.zeros((self.max_buffer_size, self.n_pixels)).astype('float32')

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

    def dis_fake_loss(self, x):
        return F.sum((x-0.1)**2/np.prod(x.data.shape))

    def dis_real_loss(self, x):
        return F.sum((x-0.9)**2/np.prod(x.data.shape))

    def gen_loss(self, x):
        return F.sum((x-0.9)**2/np.prod(x.data.shape))

    def cycle_loss(self, x, y):
        return F.mean_absolute_error(x, y)

    def train(self, n_epochs, batch_iter):
        print('Start training CycleGAN')
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