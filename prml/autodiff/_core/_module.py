from contextlib import contextmanager
from prml.autodiff._core._array import Array


class Module(object):

    def __init__(self):
        self._is_setting_parameter = False
        self.parameter = {}

    @property
    def is_setting_parameter(self):
        return getattr(self, "_is_setting_parameter", False)

    @contextmanager
    def set_parameter(self):
        prev_scope = self._is_setting_parameter
        object.__setattr__(self, "_is_setting_parameter", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_is_setting_parameter", prev_scope)

    def __setattr__(self, key, value):
        if self.is_setting_parameter:
            if isinstance(value, Array):
                self.parameter[self.__class__.__name__ + "." + key] = value
            elif isinstance(value, Module):
                for name, param in value.parameter.items():
                    actualkey = ".".join([self.__class__.__name__, key, name])
                    self.parameter[actualkey] = param

        object.__setattr__(self, key, value)

    def clear(self):
        for param in self.parameter.values():
            param.cleargrad()
