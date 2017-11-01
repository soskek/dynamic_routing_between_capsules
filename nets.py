import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def _augmentation(x):
    xp = cuda.get_array_module(x)
    MAX_SHIFT = 2
    batchsize, ch, h, w = x.shape
    h_shift, w_shift = xp.random.randint(-MAX_SHIFT, MAX_SHIFT + 1, size=2)
    a_h_sl = slice(max(0, h_shift), h_shift + h)
    a_w_sl = slice(max(0, w_shift), w_shift + w)
    x_h_sl = slice(max(0, - h_shift), - h_shift + h)
    x_w_sl = slice(max(0, - w_shift), - w_shift + w)
    a = xp.zeros(x.shape)
    a[:, :, a_h_sl, a_w_sl] = x[:, :, x_h_sl, x_w_sl]
    return a.astype(x.dtype)


def _count_params(m):
    print('# of params', sum(param.size for param in m.params()))


def squash(ss):
    ss_norm2 = F.sum(ss ** 2, axis=1, keepdims=True)
    """
    # ss_norm2 = F.broadcast_to(ss_norm2, ss.shape)
    # vs = ss_norm2 / (1. + ss_norm2) * ss / F.sqrt(ss_norm2): naive
    """
    norm_div_1pnorm2 = F.sqrt(ss_norm2) / (1. + ss_norm2)
    norm_div_1pnorm2 = F.broadcast_to(norm_div_1pnorm2, ss.shape)
    vs = norm_div_1pnorm2 * ss  # :efficient
    # (batchsize, 16, 10)
    return vs


def get_norm(vs):
    return F.sqrt(F.sum(vs ** 2, axis=1))


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

        self.results['loss'].append(self.loss.data * t.shape[0])
        self.results['correct'].append(self.calculate_correct(out, t))
        self.results['N'] += t.shape[0]
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
            Cs = F.broadcast_to(cs[:, None], Preds.shape)
            assert(Cs.shape == (batchsize, 16, 10, 32, 6 * 6))
            ss = F.sum(Cs * Preds, axis=(3, 4))
            vs = squash(ss)
            assert(vs.shape == (batchsize, 16, 10))

            if i_iter != n_iterations - 1:
                Vs = F.broadcast_to(vs[:, :, :, None, None], Preds.shape)
                assert(Vs.shape == (batchsize, 16, 10, 32, 6 * 6))
                bs = bs + F.sum(Vs * Preds, axis=1)
                assert(bs.shape == (batchsize, 10, 32, 6 * 6))

        vs_norm = get_norm(vs)
        return vs_norm, vs

    def calculate_loss(self, v, t):
        xp = self.xp
        batchsize = t.shape[0]
        I = xp.arange(batchsize)

        T = xp.zeros(v.shape, dtype='f')
        T[I, t] = 1.
        m = xp.full(v.shape, 0.1, dtype='f')
        m[I, t] = 0.9

        loss = T * F.relu(m - v) ** 2 + 0.5 * (1. - T) * F.relu(v - m) ** 2
        return F.sum(loss) / batchsize

    def calculate_correct(self, v, t):
        return (self.xp.argmax(v.data, axis=1) == t).sum()
