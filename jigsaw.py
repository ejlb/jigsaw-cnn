import chainer

from chainer import functions as F
from chainer import links as L


class Jigsaw(chainer.Chain):
    """ Siamese jigsaw CNN for self-supervised learning on image patches. The patch
        representations come from an Alexnet-type network adapted to take 64x64 crops """

    def __init__(self):
        super(Jigsaw, self).__init__(
            conv1=L.Convolution2D(3, 96, 7, stride=2, pad=2),
            bn1=L.BatchNormalization(96),

            conv2=L.Convolution2D(96, 256, 5, pad=1),
            bn2=L.BatchNormalization(256),

            conv3=L.Convolution2D(256, 384, 3, pad=1),
            conv4=L.Convolution2D(384, 384, 3, pad=1),
            conv5=L.Convolution2D(384, 256, 3, pad=1),

            fc6=L.Linear(256 * 3 * 3, 512),  # representation for one patch
            fc7=L.Linear(512 * 9, 4096),     # concat of 9 x 512 patch representations
            fc8=L.Linear(4096, 64),
        )

    def patch_representation(self, x):
        """ Input is a batch of 64 x 64 image patches """
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(self.bn1(h), 3, stride=2)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(self.bn2(h), 3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)

        """ This is concatenated across 9 patches to get 4608 representation """
        h = F.relu(self.fc6(h))

        return h


    def __call__(self, x, t):
        """ Input is a batch of patches of shape:
                (patches, batches, channels, width, height)

            Jigsaw CNN uses:
                patches = 9
                width = 64
                height = 64
        """

        patch_representations = []

        """ Split into patch batches of shape (batches, channels, height, width) and calculate
            alexnet representation for each patch """
        for patch_batch in F.split_axis(x, x.data.shape[0], 0):
            # drop patch axis for alexnet
            h = F.reshape(patch_batch, x.data.shape[1:])
            h = self.patch_representation(h)
            patch_representations.append(h)

        """ Join along the representation axis """
        h = F.concat(patch_representations, axis=1)

        h = F.relu(self.fc7(h))
        h = self.fc8(h)

        """ Loss is a prediction of which permutation we applied to these patches """
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
