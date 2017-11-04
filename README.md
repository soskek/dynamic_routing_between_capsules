# Dynamic Routing Between Capsules

Chainer implementation of CapsNet for MNIST.

For the detail, see [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf), Sara Sabour, Nicholas Frosst, Geoffrey E Hinton, NIPS 2017.

```
python -u train.py -g 0
```

Test accuracy of a trained model reached 99.60%.
The paper does not provide detailed information about initialization and optimization, so the performance might not reach that in the paper. For alleviating those issues, I replaced relu with leaky relu with very small slope (0.05). The modified model achieved 99.65% (i.e. error rate is 0.35%), as the paper reported.

TODO

- Add reconstruction loss
- Find better hyperparameters
