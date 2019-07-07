from contextlib import contextmanager
from prml.autodiff._core._array import Array


class Module(object):

    def __init__(self):
        self._is_initializing = False
        self.parameter = {}

    @property
    def is_initializing(self):
        return getattr(self, "_is_initializing", False)

    @contextmanager
    def initialize(self):
        prev_scope = self._is_initializing
        object.__setattr__(self, "_is_initializing", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_is_initializing", prev_scope)

    def __setattr__(self, key, value):
        if self.is_initializing:
            if isinstance(value, Array):
                value._parent = None
                self.parameter[self.__class__.__name__ + "." + key] = value
            elif isinstance(value, Module):
                for name, param in value.parameter.items():
                    actualkey = ".".join([self.__class__.__name__, key, name])
                    self.parameter[actualkey] = param

        object.__setattr__(self, key, value)

    def clear(self):
        for param in self.parameter.values():
            param.cleargrad()
