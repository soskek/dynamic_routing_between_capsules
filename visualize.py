from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import json
import numpy as np

import chainer
from chainer.dataset.convert import concat_examples
from chainer import serializers

import nets


def save_images(x, filename, x_raw=None):
    x = np.array(x.tolist(), np.float32)
    width = x.shape[0]

    if x_raw is not None:
        x_raw = np.array(x_raw.tolist(), np.float32)
        fig, ax = plt.subplots(2, width, figsize=(1 * width, 2))  # , dpi=20)
        xs = np.concatenate([x_raw, x], axis=0)
    else:
        fig, ax = plt.subplots(1, width, figsize=(1 * width, 1))  # , dpi=20)
        xs = x
    for ai, xi in zip(ax.ravel(), xs):
        ai.set_xticklabels([])
        ai.set_yticklabels([])
        ai.set_axis_off()
        ai.imshow(xi.reshape(28, 28), cmap='Blues_r', vmin=0., vmax=1.)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.)
    fig.savefig(filename, bbox_inches='tight', pad=0.)
    plt.clf()
    plt.close('all')


def visualize_reconstruction(model, x, t, filename='vis.png'):
    vs_norm, vs = model.output(x)
    x_recon = model.reconstruct(vs, t)
    save_images(x_recon.data[:, 0, :, :],
                filename,
                x_raw=x[:, 0, :, :])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CapsNet: MNIST reconstruction')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--load')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    model = nets.CapsNet(use_reconstruction=True)
    serializers.load_npz(args.load, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    train, test = chainer.datasets.get_mnist(ndim=3)

    batch = test[:10]
    x, t = concat_examples(batch, args.gpu)
    print('encode, reconstruct, visualize')
    visualize_reconstruction(model, x, t)
