import chainer

from chainer import functions as F
from chainer import links as L


class CNN(chainer.Chain):
    """ Alexnet-type network adapted to take 64x64 crops for jigsaw cnn. Stops at first dense
        layer. Dense layer representations are aggregated in the jigsaw CNN """

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 96, 7, stride=2, pad=2),
            bn1=L.BatchNormalization(96),
            conv2=L.Convolution2D(96, 256, 5, pad=1),
            bn2=L.BatchNormalization(256),
            conv3=L.Convolution2D(256, 384, 3, pad=1),
            conv4=L.Convolution2D(384, 384, 3, pad=1),
            conv5=L.Convolution2D(384, 256, 3, pad=1),
            fc6=L.Linear(256 * 3 * 3, 512),
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(self.bn1(h), 3, stride=2)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(self.bn2(h), 3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)

        """ This is concatenated in the jigsaw CNN across 9 patches to get 4608 representation """
        h = F.relu(self.fc6(h))

        return h
