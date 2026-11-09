#!/usr/bin/env python3
"""
CIFAR10 specific classifier with ResNet backbone
"""

from classifier.classification_modules import MRClassifier
from classifier.classification_modules.classification_models import PreTrainedResNet


class CIFAR10Classifier(MRClassifier):
    """CIFAR10 specific classifier with ResNet backbone"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize the classifier model
        self.classifier = PreTrainedResNet(config)
