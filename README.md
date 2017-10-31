# Dynamic Routing Between Capsules

Chainer implementation of CapsNet for MNIST.

For the detail, see [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf), Sara Sabour, Nicholas Frosst, Geoffrey E Hinton, NIPS 2017.

```
python -u train.py -g 0
```

## WIP

- Add reconstruction loss
- Check the number of parameters; the paper says CapsNet has 11.36M parameters, but really?
- Find better hyperparamters
    - initialization
    - minibatch size
    - decay rate of Adam's alpha
- Experiment on MultiMNIST, Cifar10, smallNORB, SVHN
