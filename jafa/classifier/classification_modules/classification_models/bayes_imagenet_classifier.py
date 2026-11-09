from classifier.classification_modules.bayes_enforcing_module import BayesDistillClassifier
from classifier.classification_modules.pretrained_resnet import PreTrainedResNet


class BayesImageNetClassifier(BayesDistillClassifier):
    """
    Bayes-Enforcing ImageNet Classifier
    """
    def __init__(self, config):
        super().__init__(config)
        self.classifier = PreTrainedResNet(
            config=config
        )

