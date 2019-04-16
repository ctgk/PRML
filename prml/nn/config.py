import numpy as np


class Config(object):
    __dtype = np.float32
    __is_updating_bn = False
    __available_dtypes = (np.float16, np.float32, np.float64)
    __enable_backprop = True

    @property
    def dtype(self):
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype in self.__available_dtypes:
            self.__dtype = dtype
        else:
            raise ValueError

    @property
    def is_updating_bn(self):
        return self.__is_updating_bn

    @is_updating_bn.setter
    def is_updating_bn(self, flag):
        assert(isinstance(flag, bool))
        self.__is_updating_bn = flag

    @property
    def enable_backprop(self):
        return self.__enable_backprop

    @enable_backprop.setter
    def enable_backprop(self, flag):
        assert(isinstance(flag, bool))
        self.__enable_backprop = flag


config = Config()
