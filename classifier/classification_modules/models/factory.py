from copy import deepcopy


def build_model(config):
    """
    Build a model instance based on config.
    - image input -> PreTrainedResNet
    - kspace input -> kSpaceNet or kspaceMRNet (via kspace model utils)
    """
    input_type = getattr(config, "input_type", "kspace")
    if input_type == "kspace":
        from ..model_utils import get_model as get_kspace_model

        return get_kspace_model(config=deepcopy(config))

    from ..pretrained_resnet import (
        PreTrainedResNet,
    )

    return PreTrainedResNet(config)
