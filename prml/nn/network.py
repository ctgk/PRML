from prml.tensor.parameter import Parameter


class Network(object):

    def __init__(self, **kwargs):
        self.params = {}
        for key in kwargs:
            param = Parameter(kwargs[key])
            self.params[key] = param
            object.__setattr__(self, key, param)

    def cleargrads(self):
        for key in self.params:
            self.params[key].cleargrad()
