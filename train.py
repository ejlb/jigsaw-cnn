# -1 for cpu
# save option
# number of test / train images
# visualisations

# number of patches
# benchmark against something (imagenet?)

import argparse

import chainer

from chainer import optimizers
from chainer.optimizer import WeightDecay
from chainer.training import extensions

import dataset

from jigsaw import Jigsaw


def parse_args():
    parser = argparse.ArgumentParser(description='Train jigsaw CNN.')

    parser.add_argument('--batch-size', type=int, default=192,
                        help='images per batch')
    parser.add_argument('--epochs', type=int, default=10,
                        help='maximum number of epochs')
    parser.add_argument('--cpu-procs', type=int, default=10,
                        help='CPU processes to use for batch loading')
    parser.add_argument('--iter-trigger', type=int, default=380,
                        help='Iteration multiples that trigger logging')
    parser.add_argument('--mean', type=int, default=183,
                        help='value to subtract from input images')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu id to run on (-1 for cpu)')
    parser.add_argument('--save', help='directory in which to save epoch snapshots.')

    parser.add_argument('train_glob', help='path to directory of images for training')
    parser.add_argument('test_glob', help='path to directory of images for test')

    return parser.parse_args()


args = parse_args()

print args
print 'loading data ...'

train_dataset = dataset.PatchDataset(args.train_glob, args.mean)
test_dataset = dataset.PatchDataset(args.test_glob, args.mean)

print 'train:%d test:%d' % (len(train_dataset), len(test_dataset))

print 'building ...'

jigsaw = Jigsaw()
jigsaw.to_gpu(args.gpu)

optimizer = optimizers.Adam()
optimizer.setup(jigsaw)
optimizer.add_hook(WeightDecay(0.0001))

train_iterator = chainer.iterators.MultiprocessIterator(
    dataset=train_dataset,
    batch_size=args.batch_size,
    n_processes=args.cpu_procs,
)

test_iterator = chainer.iterators.MultiprocessIterator(
    dataset=test_dataset,
    batch_size=args.batch_size,
    n_processes=args.cpu_procs,
    shuffle=False,
    repeat=False,
)

updater = chainer.training.StandardUpdater(
    iterator=train_iterator,
    optimizer=optimizer,
    device=args.gpu,
)


trainer = chainer.training.Trainer(
    updater,
    (args.epochs, 'epoch'),
)

# out=args.out_dir,
trainer.extend(extensions.LogReport(trigger=(args.iter_trigger, 'iteration')))

trainer.extend(extensions.PrintReport([
    'epoch',
    'iteration',
    'main/loss',
    'main/accuracy',
    'validation/main/loss',
    'validation/main/accuracy',
]))

trainer.extend(extensions.Evaluator(
    iterator=test_iterator,
    target=jigsaw,
    device=args.gpu,
))

trainer.extend(extensions.ProgressBar(update_interval=1, bar_length=50))


"""
trainer.extend(extensions.snapshot(filename=constants.TRAINER_SAVE_NAME))

trainer.extend(extensions.snapshot_object(
    target=triplet_model,
    filename=constants.MODEL_SAVE_NAME,
))
"""

print 'training ...'
trainer.run()
