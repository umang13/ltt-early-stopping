from classifier.classification_modules.bayes_enforcing_module import BayesDistillClassifier
from classifier.classification_modules.pretrained_resnet import PreTrainedResNet


class BayesCIFAR10Classifier(BayesDistillClassifier):
    """
    Bayes-Enforcing CIFAR10 Classifier
    """
    def __init__(self, config):
        super().__init__(config)
        self.classifier = PreTrainedResNet(
            config=config
        )

