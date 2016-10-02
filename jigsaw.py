import chainer

from chainer import functions as F
from chainer import links as L


class Jigsaw(chainer.Chain):
    """ Siamese jigsaw CNN for self-supervised learning """

    def __init__(self, alexnet):
        super(Jigsaw, self).__init__(
            fc7=L.Linear(512 * 9, 4096),  # concat of 9 x 512 patch representations
            fc8=L.Linear(4096, 100),
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

        """ Split into patch batches of shape (batches, channels, height, width) and calculate
            alexnet representation for each patch """
        for patch_batch in F.split_axis(x, x.data.shape[0], 0):
            # drop patch axis for alexnet
            h = F.reshape(patch_batch, x.data.shape[1:])
            h = self.alexnet(h)
            patch_representations.append(h)

        """ Join along the representation axis """
        h = F.concat(patch_representations, axis=1)

        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h))

        """ Loss is a prediction of which permutation we applied to these patches """
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
