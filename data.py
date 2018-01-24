import chainer
from chainer.datasets import TupleDataset
import numpy as np

def select_data(dataset, n_train=100, n_test=100, n_dim=1, with_label=True, classes = None):
    """

    :param n_train: nr of training examples per class
    :param n_test: nr of test examples per class
    :param n_dim: 1 or 3 (for convolutional input)
    :param with_label: whether or not to also provide labels
    :param classes: if not None, then it selects only those classes, e.g. [0, 1]
    :return:
    """

    train_data, test_data = dataset(ndim=n_dim, withlabel=True)

    for d in range(2):

        if d == 0:
            data = train_data._datasets[0]
            labels = train_data._datasets[1]
            n = n_train
        else:
            data = test_data._datasets[0]
            labels = test_data._datasets[1]
            n = n_test

        if not classes:
            c = set(labels)
        else:
            c = classes
        n_classes = len(c)

        for i in range(n_classes):
            lidx = np.where(labels == c[i])[0][:n]
            if i == 0:
                idx = lidx
            else:
                idx = np.hstack([idx, lidx])

        if with_label:
            L = np.concatenate([i * np.ones(n) for i in np.arange(n_classes)]).astype('int32')

            if d == 0:
                train_data = TupleDataset(data[idx], L)
            else:
                test_data = TupleDataset(data[idx], L)
        else:
            if d == 0:
                train_data = data[idx]
            else:
                test_data = data[idx]

    return train_data, test_data


def get_cifar10(n_train=100, n_test=100, n_dim=1, with_label=True, classes = None):
    """

    :param n_train: nr of training examples per class
    :param n_test: nr of test examples per class
    :param n_dim: 1 or 3 (for convolutional input)
    :param with_label: whether or not to also provide labels
    :param classes: if not None, then it selects only those classes, e.g. [0, 1]
    :return:
    """

    return select_data(chainer.datasets.get_cifar10, n_train=n_train, n_test=n_test, n_dim=n_dim, with_label=with_label, classes=classes )

def get_mnist(n_train=100, n_test=100, n_dim=1, with_label=True, classes = None):
    """

    :param n_train: nr of training examples per class
    :param n_test: nr of test examples per class
    :param n_dim: 1 or 3 (for convolutional input)
    :param with_label: whether or not to also provide labels
    :param classes: if not None, then it selects only those classes, e.g. [0, 1]
    :return:
    """

    return select_data(chainer.datasets.get_mnist, n_train=n_train, n_test=n_test, n_dim=n_dim, with_label=with_label, classes=classes )

def pair_datasets(dataset_a, dataset_b):
    n = min(len(dataset_a), len(dataset_b))
    return TupleDataset(dataset_a[:n], dataset_b[:n])