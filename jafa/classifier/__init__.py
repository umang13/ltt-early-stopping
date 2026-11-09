__all__ = ["ARMSModule", "BayesDistillModule"]


def __getattr__(name):
    if name == "ARMSModule":
        from .classification_modules.modules.arms_module import ARMSModule

        return ARMSModule
    if name == "BayesDistillModule":
        from .classification_modules.modules.bayes_distill_module import BayesDistillModule

        return BayesDistillModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
