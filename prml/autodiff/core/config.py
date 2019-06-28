import numpy as np


class Config(object):
    _dtype = np.float32
    _available_dtypes = (np.float16, np.float32, np.float64)
    _enable_backprop = True

    def __repr__(self):
        return (
            "autodiff.config("
            f"dtype={self._dtype}, "
            f"enable_backprop={self._enable_backprop})"
        )

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype in self._available_dtypes:
            self._dtype = dtype
        else:
            raise ValueError

    @property
    def enable_backprop(self):
        return self._enable_backprop

    @enable_backprop.setter
    def enable_backprop(self, flag):
        if not isinstance(flag, bool):
            raise TypeError
        self._enable_backprop = flag


config = Config()
