import numpy as np
from chainer import optimizers

from models.networks import BasicDiscriminatorNetwork as DiscriminatorNetwork
from models.networks import BasicGeneratorNetwork as GeneratorNetwork
# Basic GAN implementation from Assignment6.ipynb

class GANModel():

    def __init__(self, n_pixels, g_hidden, d_hidden,d_learning_rate=0.01, g_learning_rate=0.05):
        g_input_size = n_pixels
        g_hidden_size = g_hidden
        g_output_size = n_pixels

        d_input_size = g_output_size
        d_hidden_size = d_hidden
        d_output_size = 1

        self.D = DiscriminatorNetwork(d_hidden_size, d_output_size, n_input=d_input_size)
        self.G = GeneratorNetwork(g_hidden_size, g_output_size, n_input=g_input_size)

        self.d_optimizer = optimizers.SGD(lr=d_learning_rate)
        self.g_optimizer = optimizers.SGD(lr=g_learning_rate)
        self.d_optimizer.setup(self.D)
        self.g_optimizer.setup(self.G)

    def train(self, d_input_iter, g_sampler, lossfun, n_epochs = 100, d_steps = 1000, g_steps = 1000, minibatch_size = 1):

        print('start training loop')

        for epoch in range(0, n_epochs):

            if epoch % int(n_epochs/10) == 0:
                print('epoch: {0}/{1}'.format(epoch,n_epochs))

            self.d_optimizer.new_epoch()
            self.g_optimizer.new_epoch()

            for d_index in range(0, d_steps):

                # Train Discriminator
                self.D.cleargrads()

                # Train on real data (discriminator learns what real pictures look like)
                d_real_data = d_input_iter.next()
                d_real_pred = self.D(np.array(d_real_data))
                d_real_error = lossfun(d_real_pred, np.ones((minibatch_size,1), dtype='float32'))
                d_real_error.backward()

                # train on fake data (discriminator learns what fake pictures look like)
                d_gen_input = g_sampler(minibatch_size)
                d_fake_data = self.G(d_gen_input)
                d_fake_pred = self.D(d_fake_data)
                d_fake_error = lossfun(d_fake_pred, np.zeros((minibatch_size,1), dtype='float32'))
                d_fake_error.backward()
                self.d_optimizer.update()

            for g_index in range(0, g_steps):

                # train G on D's response
                self.G.cleargrads()

                gen_input = g_sampler(minibatch_size)
                g_fake_data = self.G(gen_input) # generate a fake picture
                dg_fake_pred = self.D(g_fake_data) # to what degree does the discriminator think the picture is true
                g_error = lossfun(dg_fake_pred, np.ones((minibatch_size,1), dtype='float32')) # loss is discriminator-estimated 'fakeness'
                g_error.backward()
                self.g_optimizer.update()

        print('done!')
