import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


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


class CapsNet(chainer.Chain):

    def __init__(self):
        super(CapsNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 256, ksize=9, stride=1)
            self.conv2 = L.Convolution2D(256, 256, ksize=9, stride=2)
            self.Ws = chainer.ChainList(
                *[L.Convolution2D(8, 16 * 10, ksize=1, stride=1)
                  for i in range(32)])

            # TODO: use NDconv
        #"""
        for name, param in self.namedparams():
            if param.ndim != 1:
                # This initialization is applied only for weight matrices
                param.data[...] = np.random.uniform(
                    -0.01, 0.01, param.data.shape)
        #"""

    def __call__(self, x, t):
        #x = _augmentation(x)
        out, _ = self.output(x)
        self.loss = self.calculate_loss(out, t)
        self.accuracy = self.calculate_accuracy(out, t)
        print(self.loss, self.accuracy)
        return self.loss

    def evaluate(self, x, t):
        out, _ = self.output(x)
        self.loss = self.calculate_loss(out, t)
        self.accuracy = self.calculate_accuracy(out, t)
        return self.loss.data, self.accuracy

    def output(self, x):
        batchsize = x.shape[0]
        n_iterations = 3

        h1 = F.relu(self.conv1(x))
        primary_caps_concat = self.conv2(h1)  # relu required?
        primary_caps = F.split_axis(primary_caps_concat, 32, axis=1)

        # TODO: do not average it
        Preds = []
        for i in range(32):
            pred = self.Ws[i](primary_caps[i])
            assert(pred.shape[1] == 160)
            assert(pred.shape[2] == 6)
            assert(pred.shape[3] == 6)
            Pred = pred.reshape((batchsize, 10, 16, 6 * 6))
            #Pred = pred.reshape((batchsize, 160, 6 * 6))
            #Pred = Pred.reshape((batchsize, 10, 16, 6 * 6))
            Preds.append(Pred)

        bs = [self.xp.zeros((batchsize, 10, 1, 6 * 6), dtype='f')] * 32
        for i_iter in range(n_iterations):
            ss = self.xp.zeros((batchsize, 10, 16), dtype='f')
            for i in range(32):
                b = bs[i]
                c = F.softmax(b, axis=1)
                C = F.broadcast_to(c, (batchsize, 10, 16, 6 * 6))
                ss = ss + F.sum(C * Preds[i], axis=3)
            ss_norm = F.sum(ss ** 2, axis=2)
            Ss_norm = F.broadcast_to(ss_norm[:, :, None], ss.shape)
            vs = Ss_norm / (1. + Ss_norm) * ss / (Ss_norm ** 0.5)
            # (batchsize, 10, 16)

            if i_iter != n_iterations - 1:
                for i in range(32):
                    Vs = F.broadcast_to(vs[:, :, :, None], Preds[i].shape)
                    bs[i] = bs[i] + F.sum(Vs * Preds[i], axis=2, keepdims=True)
                    # F.sum(vs * Preds[i], axis=2)

        vs_norm = F.sum(vs ** 2, axis=2) ** 0.5
        return vs_norm, vs

    def calculate_loss(self, v, t):
        T = self.xp.zeros(v.shape, dtype='f')
        T[t] = 1.
        m = self.xp.full(v.shape, 0.1, dtype='f')
        m[t] = 0.9
        loss = T * F.relu(m - v) ** 2 + 0.5 * (1. - T) * F.relu(v - m) ** 2
        # return F.sum(loss)
        return F.mean(loss) * 10

    def calculate_accuracy(self, v, t):
        return (self.xp.argmax(v.data, axis=1) == t).mean()
