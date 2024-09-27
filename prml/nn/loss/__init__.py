from prml.nn.loss.kl import kl_divergence
from prml.nn.loss.sigmoid_cross_entropy import sigmoid_cross_entropy
from prml.nn.loss.softmax_cross_entropy import softmax_cross_entropy


_functions = [kl_divergence, sigmoid_cross_entropy, softmax_cross_entropy]


__all__ = [_f.__name__ for _f in _functions]


del _functions
