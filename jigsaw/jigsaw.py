import chainer

from chainer import functions as F
from chainer import links as L
from chainer import reporter


def squeeze(x):
    return F.reshape(x, [di for di in x.data.shape if di != 1])


class Jigsaw(chainer.Chain):
    """ Siamese jigsaw CNN for self-supervised learning on image patches. The patch
        representations come from an Alexnet-type network adapted to take 64x64 crops
    """

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
            fc8=L.Linear(4096, 100),
        )

    def patch_representation(self, x, dense_layer=True):
        """ Calculate representation for a batch of 64 x 64 image patches. """

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(self.bn1(h), 3, stride=2)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(self.bn2(h), 3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        if dense_layer:
            h = F.relu(self.fc6(h))

        return h

    def jigsaw_representation(self, x):
        """ Calculate representation for batch of images """

        n_batch = x.data.shape[0]
        n_patch = x.data.shape[1]

        # merge batch and patch axis for batch representation calculation
        h = F.reshape(x, (
            n_batch * n_patch,
            x.data.shape[2],
            x.data.shape[3],
            x.data.shape[4],
        ))

        h = self.patch_representation(h)

        # split into batch and patch axis
        h = F.reshape(h, (n_batch, n_patch, -1))

        h = F.relu(self.fc7(h))
        h = self.fc8(h)

        return h

    def __call__(self, x, t):
        """ Input is a batch of patches of shape:
                (batches, patches, channels, width, height)

            Jigsaw CNN uses:
                patches = 9
                width = 64
                height = 64
        """

        # drop join axis from chainer dataset abstraction
        x = squeeze(x)
        t = squeeze(t)

        h = self.jigsaw_representation(x)

        # loss is a prediction of which permutation we applied to these patches
        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)

        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)

        return loss
