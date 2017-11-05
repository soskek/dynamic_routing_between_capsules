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


def save_images(xs, filename):
    width = xs[0].shape[0]
    height = len(xs)

    xs = [np.array(x.tolist(), np.float32) for x in xs]
    fig, ax = plt.subplots(height, width, figsize=(1 * width, height))
    xs = np.concatenate(xs, axis=0)

    for ai, xi in zip(ax.ravel(), xs):
        ai.set_xticklabels([])
        ai.set_yticklabels([])
        ai.set_axis_off()
        ai.imshow(xi.reshape(28, 28), cmap='Blues_r', vmin=0., vmax=1.)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.)
    fig.savefig(filename, bbox_inches='tight', pad=0.)
    plt.clf()
    plt.close('all')


def visualize_reconstruction(model, x, t, filename='vis.png'):
    vs_norm, vs = model.output(x)
    x_recon = model.reconstruct(vs, t)
    save_images([x, x_recon.data],
                filename)


def visualize_reconstruction_all(model, x, t, filename='visall.png'):
    t_list = []
    x_recon_list = []
    vs_norm, vs = model.output(x)
    for i in range(10):
        pseudo_t = model.xp.full(t.shape, i).astype('i')
        x_recon = model.reconstruct(vs, pseudo_t).data
        t_list.append(pseudo_t)
        x_recon_list.append(x_recon)
    save_images([x] + x_recon_list,
                filename)


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

    batch = test[50:60]
    x, t = concat_examples(batch, args.gpu)

    visualize_reconstruction(model, x, t)
    visualize_reconstruction_all(model, x, t)
