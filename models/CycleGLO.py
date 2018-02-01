from chainer import optimizers, Variable
import chainer.functions as F
import numpy as np
import matplotlib.pyplot as plt

from models.networks import BasicGLOGeneratorNetwork as GenerativeNetwork
from models.GLO import initialZ, project_z_to_ball

class CycleGLO():

    def __init__(self, alpha, beta, lambda1, lambda2, n_pixels, n_vecs, dataset, learning_rate_decay, learning_rate_interval, g_hidden, d_hidden, max_buffer_size):
        self.max_buffer_size = max_buffer_size

        g_input_size = n_pixels
        g_hidden_size = g_hidden
        g_output_size = n_pixels

        z_output_size = n_pixels

        self.dataset = dataset

        self.g = GenerativeNetwork(g_hidden_size, g_output_size, n_input=g_input_size)
        self.f = GenerativeNetwork(g_hidden_size, g_output_size, n_input=g_input_size)

        self.zy = initialZ(len(dataset), z_output_size, dataset, method='random')
        self.zx = initialZ(len(dataset), z_output_size, dataset, method='random')

        self.opt_g = optimizers.Adam(alpha=alpha, beta1=beta)
        self.opt_f = optimizers.Adam(alpha=alpha, beta1=beta)
        self.opt_zx = optimizers.Adam(alpha=alpha, beta1=beta)
        self.opt_zy = optimizers.Adam(alpha=alpha, beta1=beta)

        self.opt_g.setup(self.g)
        self.opt_f.setup(self.f)
        self.opt_zx.setup(self.g.z)
        self.opt_zy.setup(self.f.z)

        self.n_pixels = n_pixels
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_interval = learning_rate_interval

        # representation space gaussian distr to be learnt from learnable latent vectors Z
        self.xrspace_mean = []
        self.xrspace_std = []
        self.yrspace_mean = []
        self.yrspace_std = []

        self.buffer = {}
        self.buffer['x'] = np.zeros((self.max_buffer_size, self.n_pixels)).astype('float32')
        self.buffer['y'] = np.zeros((self.max_buffer_size, self.n_pixels)).astype('float32')

    def getAndUpdateBuffer(self, buffer, data, epoch):
        if epoch < self.max_buffer_size:
            self.buffer[buffer][epoch] = data[0]
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

    def train(self,lossfun, n_epochs=100):
        print('Start training CycleGLO')
        losses = []
        for epoch in range(n_epochs):
            print(epoch)
            self.opt_g.new_epoch()
            self.opt_f.new_epoch()
            self.opt_zx.new_epoch()
            self.opt_zy.new_epoch()
            for i in range(len(self.dataset)):
                x = self.dataset[i][0]
                y = self.dataset[i][1]
                #print(x, y)
                x, y = Variable(x), Variable(y)
                #print(x.shape, y.shape)

                self.g.cleargrads()
                self.f.cleargrads()
                self.g.z.cleargrads()
                self.f.z.cleargrads()

                xy = self.g(self.zx[i])
                yx = self.f(self.zy[i])
                #print(xy, yx)
                yxy = self.g(yx.data)
                xyx = self.f(xy.data)
                #print(yxy, xyx)
                xy_copy = Variable(self.getAndUpdateBuffer('x', xy.data, epoch))
                yx_copy = Variable(self.getAndUpdateBuffer('y', yx.data, epoch))
                #print(yx_copy.shape)
                x_loss = lossfun(yx_copy, x.reshape((1,self.n_pixels)))
                y_loss = lossfun(xy_copy, y.reshape((1,self.n_pixels)))

                g_loss = lossfun(xy, y.reshape((1,self.n_pixels)))
                f_loss = lossfun(yx, x.reshape((1,self.n_pixels)))

                cycle_x_loss = lossfun(xyx, x.reshape((1,self.n_pixels)))
                cycle_y_loss = lossfun(yxy, y.reshape((1,self.n_pixels)))
                gen_loss = self.lambda2 * g_loss + self.lambda2 * f_loss + cycle_x_loss + cycle_y_loss

                if self.learning_rate_decay > 0 and epoch % self.learning_rate_interval == 0 :
                    if self.opt_g.alpha > self.learning_rate_decay:
                        self.opt_g.alpha -= self.learning_rate_decay
                    if self.opt_f.alpha > self.learning_rate_decay:
                        self.opt_f.alpha -= self.learning_rate_decay

                x_loss.backward()
                y_loss.backward()
                self.opt_zx.update()
                self.opt_zy.update()

                #### Update
                gen_loss.backward()
                self.opt_g.update()
                self.opt_f.update()

                self.zx[i] = project_z_to_ball(self.zx[i] - 0.1 * self.g.z.z.grad)
                self.zy[i] = project_z_to_ball(self.zy[i] - 0.1 * self.f.z.z.grad)

                losses += [(x_loss, y_loss, g_loss, f_loss, cycle_x_loss, cycle_y_loss, gen_loss)]

        print('done!')
        self.xrspace_mean = np.mean(self.zx, axis=0)
        self.xrspace_std = np.std(self.zx, axis=0)

        self.yrspace_mean = np.mean(self.zy, axis=0)
        self.yrspace_std = np.std(self.zy, axis=0)
        print(self.xrspace_mean, self.xrspace_std, self.yrspace_mean, self.yrspace_std)

        to_plot = [l[-1].data for l in losses]
        x_axis = list(range(n_epochs))
        plt.plot(to_plot)
        plt.title('Loss per epoch of CycleGAN')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        print('last loss:',to_plot[-1])
        plt.show()