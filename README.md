# Jigsaw CNN

Work in progress

A chainer implementation of self-supervised [jigsaw CNNs](https://arxiv.org/abs/1603.09246). The authors have published their [caffe implementation](https://github.com/MehdiNoroozi/JigsawPuzzleSolver)

Training could be made faster by precalculating batches. To identify an `n-permutation` we only need `n-1` elements so I've made the task harder by randomly zero'ing one of the patches
