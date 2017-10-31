import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def _augmentation(x):
    MAX_SHIFT = 2
    batchsize, channel, h, w = x.shape
    xp = cuda.get_array_module(x)
    h_shift, w_shift = xp.random.randint(0, MAX_SHIFT * 2, size=2)
    a = xp.zeros((batchsize, channel, h + MAX_SHIFT * 2, w + MAX_SHIFT * 2),
                 dtype=x.dtype)
    a[:, :, h_shift:h_shift + h, w_shift:w_shift + w] = x
    a = a[:, :, MAX_SHIFT:MAX_SHIFT + h, MAX_SHIFT:MAX_SHIFT + w]
    assert(a.shape == x.shape)
    return a


def _count_params(m):
    print('# of params', sum(param.size for param in m.params()))


init = chainer.initializers.Uniform(scale=0.05)


class CapsNet(chainer.Chain):

    def __init__(self):
        super(CapsNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 256, ksize=9, stride=1,
                                         initialW=init)
            self.conv2 = L.Convolution2D(256, 32 * 8, ksize=9, stride=2,
                                         initialW=init)
            self.Ws = chainer.ChainList(
                *[L.Convolution2D(8, 16 * 10, ksize=1, stride=1, initialW=init)
                  for i in range(32)])

            self.fc1 = L.Linear(16 * 10, 512, initialW=init)
            self.fc2 = L.Linear(512, 1024, initialW=init)
            self.fc3 = L.Linear(1024, 784, initialW=init)

        _count_params(self)
        self.results = {'N': 0, 'loss': [], 'correct': []}

    def pop_results(self):
        mean_loss = sum(self.results['loss']) / self.results['N']
        accuracy = sum(self.results['correct']) / self.results['N']
        self.results = {'N': 0, 'loss': [], 'correct': []}
        return mean_loss, accuracy

    def __call__(self, x, t):
        if chainer.config.train:
            x = _augmentation(x)
        out, _ = self.output(x)
        self.loss = self.calculate_loss(out, t)

        self.results['loss'].append(self.loss.data)
        self.results['correct'].append(self.calculate_correct(out, t))
        return self.loss

    def output(self, x):
        batchsize = x.shape[0]
        n_iterations = 3

        h1 = F.relu(self.conv1(x))
        pr_caps = F.split_axis(self.conv2(h1), 32, axis=1)

        Preds = []
        for i in range(32):
            pred = self.Ws[i](pr_caps[i])
            Pred = pred.reshape((batchsize, 16, 10, 6 * 6))
            Preds.append(Pred)
        Preds = F.stack(Preds, axis=3)
        assert(Preds.shape == (batchsize, 16, 10, 32, 6 * 6))

        bs = self.xp.zeros((batchsize, 10, 32, 6 * 6), dtype='f')
        for i_iter in range(n_iterations):
            cs = F.softmax(bs, axis=1)
            Cs = F.broadcast_to(cs[:, None], (batchsize, 16, 10, 32, 6 * 6))
            ss = F.sum(Cs * Preds, axis=(3, 4))
            ss_norm = F.sum(ss ** 2, axis=1, keepdims=True)
            ss_norm = F.broadcast_to(ss_norm, ss.shape)
            vs = ss_norm / (1. + ss_norm) * ss / (ss_norm ** 0.5)
            # (batchsize, 16, 10)

            if i_iter != n_iterations - 1:
                Vs = F.broadcast_to(vs[:, :, :, None, None], Preds.shape)
                # (batchsize, 16, 10, 32, 6 * 6)
                bs = bs + F.sum(Vs * Preds, axis=1)

        vs_norm = F.sum(vs ** 2, axis=1) ** 0.5
        return vs_norm, vs

    def calculate_loss(self, v, t):
        batchsize = t.shape[0]
        xp = self.xp
        I = xp.arange(batchsize)

        T = xp.zeros(v.shape, dtype='f')
        T[I, t] = 1.
        m = xp.full(v.shape, 0.1, dtype='f')
        m[I, t] = 0.9
        loss = T * F.relu(m - v) ** 2 + 0.5 * (1. - T) * F.relu(v - m) ** 2
        return F.sum(loss) / batchsize

    def calculate_correct(self, v, t):
        return (self.xp.argmax(v.data, axis=1) == t).sum()
