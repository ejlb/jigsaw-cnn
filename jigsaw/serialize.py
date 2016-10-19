from chainer import serializers

import jigsaw


def load_model(model_path):
    """
    Load a serialised jigsaw model from disk.

    Note this only loads the model, not the trainer.
    """

    model = jigsaw.Jigsaw()
    serializers.load_npz(model_path, model)
    return model
