import chainer

from chainer import functions as F
from chainer import links as L


class AlexNet(chainer.Chain):
    """ Alexnet adapted to take 64x64 crops for jigsaw cnn. Padding set to produce the dense
        layer size specified by the jigsaw CNN paper. Stops at first dense layer and patch
        representations are aggregated in the jigsaw CNN """

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=2, pad=4),
            conv2=L.Convolution2D(96, 256, 5, pad=3),
            conv3=L.Convolution2D(256, 384, 3, pad=1),
            conv4=L.Convolution2D(384, 384, 3, pad=1),
            conv5=L.Convolution2D(384, 256, 3, pad=1),
            fc6=L.Linear(4096, 512),  # concat this across 9 patches to get 4608 representation
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(h), 3, stride=2)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.local_response_normalization(h), 3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)

        h = F.relu(self.fc6(h))

        return h
