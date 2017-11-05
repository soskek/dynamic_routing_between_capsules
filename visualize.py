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
    # subplots with many figs are very slow
    fig, ax = plt.subplots(
        height, width, figsize=(1 * width / 2.5, height / 2.5))
    xs = np.concatenate(xs, axis=0)
    for i, (ai, xi) in enumerate(zip(ax.ravel(), xs)):
        ai.set_xticklabels([])
        ai.set_yticklabels([])
        ai.set_axis_off()
        color = 'Greens_r' if i < width else 'Blues_r'  # first line
        ai.imshow(xi.reshape(28, 28), cmap=color, vmin=0., vmax=1.)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.)
    # saving and clearing subplots with many figs are also very slow
    fig.savefig(filename, bbox_inches='tight', pad=0.)
    plt.clf()
    plt.close('all')


def visualize_reconstruction(model, x, t, filename='vis.png'):
    vs_norm, vs = model.output(x)
    x_recon = model.reconstruct(vs, t)
    save_images([x, x_recon.data],
                filename)


def visualize_reconstruction_alldigits(model, x, t, filename='vis_all.png'):
    x_recon_list = []
    vs_norm, vs = model.output(x)
    for i in range(10):
        pseudo_t = model.xp.full(t.shape, i).astype('i')
        x_recon = model.reconstruct(vs, pseudo_t).data
        x_recon_list.append(x_recon)
    save_images([x] + x_recon_list,
                filename)


def visualize_reconstruction_tweaked(model, x, t, filename='vis_tweaked.png',
                                     dim_idx=0):
    assert(0 <= dim_idx <= 15)
    x_recon_list = []
    vs_norm, vs = model.output(x)
    vs = vs.data
    for i in range(9):
        tweaked_vs = model.xp.array(vs)
        tweaked_vs[:, dim_idx, :] = (i - 4.) * 0.05  # [-0.20, 0.20]
        x_recon = model.reconstruct(tweaked_vs, t).data
        x_recon_list.append(x_recon)
    x_recon = model.reconstruct(vs, t).data
    save_images([x_recon] + x_recon_list,
                filename)


def get_samples(dataset):
    # 2 samples for each digit
    samples = []
    for i, (x, t) in enumerate(dataset):
        if t == len(samples) // 2:
            print('{}-th sample is used'.format(i))
            samples.append((x, t))
        if len(samples) >= 20:
            break
    return samples


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
    _, test = chainer.datasets.get_mnist(ndim=3)

    batch = get_samples(test)
    x, t = concat_examples(batch, args.gpu)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            visualize_reconstruction(model, x, t)
            visualize_reconstruction_alldigits(model, x, t)
            for i in range(16):
                visualize_reconstruction_tweaked(
                    model, x, t,
                    filename='vis_tweaked{}.png'.format(i),
                    dim_idx=i)
