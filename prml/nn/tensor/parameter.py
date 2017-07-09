from prml.nn.tensor.tensor import Tensor


class Parameter(Tensor):
    """
    parameter to be optimized
    """

    def __init__(self, array):
        super().__init__(array, function=None)
        self.grad = None

    def _backward(self, delta):
        if self.grad is None:
            self.grad = delta
        else:
            self.grad += delta

    def cleargrad(self):
        self.grad = None
