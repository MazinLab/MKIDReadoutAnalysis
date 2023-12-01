import numpy as np


class FunctionList(list):
    """A list of functions. It can be called like a function and returns a
    list of the results."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for function in self:
            self._check_type(function)

    def __call__(self, *args, **kwargs):
        new_args = []
        new_kwargs = []
        for index, function in enumerate(self):
            new_args.append([arg[index] if isinstance(arg, tuple) else arg for arg in args])
            new_kwargs.append({key: value[index] if isinstance(value, tuple) else value
                               for key, value in kwargs.items()})
        return [function(*new_args[ind], **new_kwargs[ind]) for ind, function in enumerate(self)]

    def append(self, x):
        self._check_type(x)
        super().append(x)

    def extend(self, iterable):
        for function in iterable:
            self._check_type(function)
        super().extend(iterable)

    def insert(self, i, x):
        self._check_type(x)
        super().insert(i, x)

    @staticmethod
    def _check_type(function):
        if not callable(function):
            raise TypeError("'{}' object is not callable".format(type(function)))


class ConcatenatedFunctionList(FunctionList):
    """A list of functions. It can be called like a function and returns a
    concatenated numpy.ndarray of the results."""
    def __call__(self, *args, **kwargs):
        return np.concatenate(super().__call__(*args, **kwargs))


class Joint:
    """A joint model that concatenates the results of the input models."""
    def __init__(self, *models):
        self.models = models

    def __getattr__(self, item):
        if not callable(getattr(self.models[0], item)):
            return np.concatenate([getattr(model, item) for model in self.models])
        else:
            return ConcatenatedFunctionList([getattr(model, item) for model in self.models])
