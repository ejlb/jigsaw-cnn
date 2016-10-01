import chainer

from chainer import functions as F
from chainer.functions import array as FA
from chainer import links as L


class Jigsaw(chainer.Chain):
    def __init__(self, alexnet):
        super(AlexNet, self).__init__(
            fc7=L.Linear(512 * 9, 4096),  # concat of 9 x 512 patch representations
            fc8=L.Linear(4096, 100),
            softmax=L.Linear(100, 64),
        )

        self.alexnet = alexnet

    def __call__(self, x, t):
        """ Input is a batch of patches of shape:
                (patches, batches, channels, width, height)

            Jigsaw CNN uses:
                patches = 9
                width = 64
                height = 64
        """

        patch_representations = []

        for patch_batch in FA.split_axis(x, x.data.shape[0], 0):
            h = self.alexnet(patch_batch)
            h = FA.reshape(h, [1] + x.data.shape[1:])  # reshape with extra dim for concat

        h = FA.concat(patch_representations, axis=0)

        h = self.relu(self.fc7(h))
        h = self.relu(self.fc7(h))

        return F.softmax_cross_entropy(self.softmax(h), t)
