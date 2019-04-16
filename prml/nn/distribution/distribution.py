from prml.nn.function import Function


class Distribution(Function):
    is_categorical = False

    def __init__(self, data=None):
        self.data = data

    def draw(self):
        self.data = self.forward()
        return self.data

    def pdf(self, x=None):
        if x is not None:
            return self._pdf(x)
        elif self.data is not None:
            return self._pdf(self.data)
        else:
            raise ValueError

    def _pdf(self, x):
        raise NotImplementedError

    def log_pdf(self, x=None):
        if x is not None:
            return self._log_pdf(x)
        elif self.data is not None:
            return self._log_pdf(self.data)
        else:
            raise ValueError

    def _log_pdf(self, *args, **kwargs):
        raise NotImplementedError
