from contextlib import contextmanager
from prml.nn.array.array import Array


class Network(object):

    def __init__(self):
        self._setting_parameter = False
        self.parameter = {}

    @property
    def setting_parameter(self):
        return getattr(self, "_setting_parameter", False)

    @contextmanager
    def set_parameter(self):
        prev_scope = self._setting_parameter
        object.__setattr__(self, "_setting_parameter", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_setting_parameter", prev_scope)

    def __setattr__(self, key, value):
        if self.setting_parameter:
            if isinstance(value, Array):
                self.parameter[self.__class__.__name__ + "." + key] = value
            elif isinstance(value, Network):
                for name, param in value.parameter.items():
                    self.parameter[self.__class__.__name__ + "." + key + "." + name] = param

        object.__setattr__(self, key, value)

    def clear(self):
        for param in self.parameter.values():
            param.cleargrad()
