from abc import ABC, abstractmethod


class BaseModels(ABC):
    def __init__(self, train, test, **kwargs):
        self.train = train
        self.test = test

    @abstractmethod
    def hyper_param_tuning(self):
        return


class ExpandingWindowModels(BaseModels):
    def __init__(self, train, test, **kwargs):
        super().__init__(train, test, **kwargs)

    def hyper_param_tuning(self):
        return

