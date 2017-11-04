from __future__ import print_function

import argparse
import json
import numpy as np

import chainer
from chainer.dataset.convert import concat_examples

import nets


def main():
    parser = argparse.ArgumentParser(description='CapsNet: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256)
    parser.add_argument('--decay', '-d', type=float, default=0.95)
    parser.add_argument('--epoch', '-e', type=int, default=500)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=789)
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # Set up a neural network to train
    np.random.seed(args.seed)
    model = nets.CapsNet()
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    np.random.seed(args.seed)
    model.xp.random.seed(args.seed)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, 100,
                                                 repeat=False, shuffle=False)

    best = 0.
    best_epoch = 0
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

            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    for batch in test_iter:
                        x, t = concat_examples(batch, args.gpu)
                        loss = model(x, t)
            mean_loss, accuracy = model.pop_results()
            print('\t\ttest mean  loss: {}, accuracy: {}'.format(
                mean_loss, accuracy))
            if accuracy > best:
                best, best_epoch = accuracy, train_iter.epoch
            optimizer.alpha *= args.decay
            optimizer.alpha = max(optimizer.alpha, 1e-5)
            print('\t\t# optimizer alpha', optimizer.alpha)
            test_iter.reset()
    print('Finish: Best accuray: {} at {} epoch'.format(best, best_epoch))


if __name__ == '__main__':
    main()
