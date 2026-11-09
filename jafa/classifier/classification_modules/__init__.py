__all__ = [
    "CIFAR10Classifier",
    "ImageNetClassifier",
    "BayesCIFAR10Classifier",
    "BayesImageNetClassifier",
    "ARMSClassifier",
    "BayesDistillArmsClassifier",
    "BayesDistillClassifier",
]


def __getattr__(name):
    if name in __all__:
        from .compat import (
            ARMSClassifier,
            BayesCIFAR10Classifier,
            BayesDistillArmsClassifier,
            BayesDistillClassifier,
            BayesImageNetClassifier,
            CIFAR10Classifier,
            ImageNetClassifier,
        )

        return {
            "CIFAR10Classifier": CIFAR10Classifier,
            "ImageNetClassifier": ImageNetClassifier,
            "BayesCIFAR10Classifier": BayesCIFAR10Classifier,
            "BayesImageNetClassifier": BayesImageNetClassifier,
            "ARMSClassifier": ARMSClassifier,
            "BayesDistillArmsClassifier": BayesDistillArmsClassifier,
            "BayesDistillClassifier": BayesDistillClassifier,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
