from prml.autodiff._core._function import _Function


class _ReLU(_Function):

    @staticmethod
    def _forward(x):
        return x.clip(min=0)

    @staticmethod
    def _backward(delta, x):
        return delta * (x > 0)


def relu(x):
    r"""
    activate input with Rectified linear unit

    .. math::

        \begin{cases}
            x &~& \mbox{if } x > 0\\
            0 &~& \mbox{if } x \le 0
        \end{cases}

    Parameters
    ----------
    x : array_like
        input

    Returns
    -------
    Array
        output
    """
    return _ReLU().forward(x)
