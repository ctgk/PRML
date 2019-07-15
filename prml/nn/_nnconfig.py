class NNConfig(object):
    _is_updating_bn = False

    @property
    def is_updating_bn(self):
        return self._is_updating_bn

    @is_updating_bn.setter
    def is_updating_bn(self, flag):
        if not isinstance(flag, bool):
            raise TypeError
        self._is_updating_bn = flag


nnconfig = NNConfig()
