# Jigsaw CNN

A chainer implementation of self-supervised [jigsaw CNNs](https://arxiv.org/abs/1603.09246). The
authors have published their [caffe
implementation](https://github.com/MehdiNoroozi/JigsawPuzzleSolver)

I've gone for maximum randomness in this implementation but you can get make training faster by
precalculating batches or decreasing the randomness of the patch selection and permutation.
