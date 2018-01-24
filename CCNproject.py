import numpy as np
from chainer import iterators
import chainer.functions as F
import matplotlib.pyplot as plt
    

import data
from models import GANModel, NoiseSampler
from models import CycleGAN

def main():
    # Read train/test data
    n_train = 100
    n_test = 100
    train_data1, test_data1 = data.get_mnist(n_train, n_test, 1, False, classes=[2])
    train_data2, test_data2 = data.get_mnist(n_train, n_test, 1, False, classes=[4])
    train_data = data.pair_datasets(train_data1, train_data2)

    # Create model
    n_pixels = len(train_data[0][0])
    g_hidden = 50
    d_hidden = 50
    d_learning_rate = 0.01
    g_learning_rate = 0.05
    #model = GANModel(n_pixels, g_hidden, d_hidden, d_learning_rate, g_learning_rate)

    alpha = 0.0002
    beta = 0.5
    lambda1 = 10.0
    lambda2 = 3.0
    learningrate_decay = 0.0
    learningrate_interval = 1000
    max_buffer_size = 25
    model = CycleGAN(alpha, beta, lambda1, lambda2, n_pixels, learningrate_decay, learningrate_interval, g_hidden, d_hidden, max_buffer_size)


    # Train model
    lossfun = F.mean_squared_error

    n_epochs = 30000
    d_steps = 1
    g_steps = 1
    minibatch_size = 1

    mu = 1
    sigma = 1
    noisefun = np.random.normal
    #g_sampler = NoiseSampler(fun=noisefun, loc=mu, scale=sigma, size=(n_pixels, 1))  # iterator over randomized noise
    #d_input_iter = iterators.SerialIterator(train_data, batch_size=minibatch_size, repeat=True, shuffle=True)  # iterator over real data
    # model.train(d_input_iter, g_sampler, lossfun, n_epochs, d_steps, g_steps, minibatch_size)

    batch_iter = iterators.MultiprocessIterator(train_data, batch_size=minibatch_size, n_processes=4)
    model.train(n_epochs, batch_iter)

    # Visualize training


    # Visualize result/test/performance
    sqrt_pixels = int(np.sqrt(n_pixels))
    #for g_index in range(0, 10):
    #    gen_input = g_sampler(1)
    #    g_fake_data = model.G(gen_input)
    #    f, axarr = plt.subplots(1, 2)
    #    axarr[0].imshow(gen_input.reshape((sqrt_pixels, sqrt_pixels)))
    #    axarr[0].set_title('noise input')
    #    axarr[1].imshow(g_fake_data.data.reshape((sqrt_pixels, sqrt_pixels)))
    #    axarr[1].set_title('generated sample')
    #    plt.show()
    print('Visualizing!')
    for input in test_data1[:5]:
        generated = model.g(input.reshape(1, n_pixels))
        #print(generated.data)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(input.reshape((sqrt_pixels, sqrt_pixels)))
        axarr[0].set_title('noise input G')
        axarr[1].imshow(generated.data.reshape((sqrt_pixels, sqrt_pixels)))
        axarr[1].set_title('generated sample')
        plt.show()

    for input in test_data2[:5]:
        generated = model.f(input.reshape(1, n_pixels))
        #print(generated.data)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(input.reshape((sqrt_pixels, sqrt_pixels)))
        axarr[0].set_title('noise input F')
        axarr[1].imshow(generated.data.reshape((sqrt_pixels, sqrt_pixels)))
        axarr[1].set_title('generated sample')
        plt.show()

if __name__ == '__main__':
    main()