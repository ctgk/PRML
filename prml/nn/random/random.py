from prml.nn.function import Function


class RandomVariable(Function):

    def __init__(self, data=None, p=None):
        if data is not None and p is not None:
            raise ValueError
        if data is not None:
            data = self._convert2array(data)
        self.data = data
        self.observed = (data is not None)
        self.p = p

    def draw(self):
        if self.observed:
            raise ValueError
        self.data = self.forward()
        return self.data

    def pdf(self, x=None):
        if x is not None:
            return self._pdf(x)
        if self.data is not None:
            return self._pdf(self.data)
        raise ValueError

    def _pdf(self, *args):
        raise NotImplementedError

    def log_pdf(self, x=None):
        if x is not None:
            return self._log_pdf(x)
        if self.data is not None:
            return self._log_pdf(self.data)
        raise ValueError

    def _log_pdf(self, *args):
        raise ValueError

    def KLqp(self):
        if self.p is None or self.data is None:
            raise ValueError
        return self._log_pdf(self.data) - self.p._log_pdf(self.data)
