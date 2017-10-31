from __future__ import print_function

import argparse
import json

import chainer
from chainer.dataset.convert import concat_examples

import nets

DECAY_LR = 0.8


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # Set up a neural network to train
    model = nets.CapsNet()
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, 100,
                                                 repeat=False, shuffle=False)

    print('TRAINING starts')
    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        x, t = concat_examples(batch, args.gpu)
        optimizer.update(model, x, t)

        # evaluation
        if train_iter.is_new_epoch:
            mean_loss, accuracy = model.pop_results()
            print('epoch {:2d}\ttrain mean loss: {}, accuracy: {}'.format(
                train_iter.epoch, mean_loss, accuracy))

            for batch in test_iter:
                x, t = concat_examples(batch, args.gpu)
                with chainer.no_backprop_mode():
                    with chainer.using_config('train', False):
                        loss = model(x, t)

            test_iter.reset()
            mean_loss, accuracy = model.pop_results()
            print('\t\ttest mean  loss: {}, accuracy: {}'.format(
                mean_loss, accuracy))
            optimizer.alpha *= DECAY_LR


if __name__ == '__main__':
    main()
