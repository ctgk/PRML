import unittest
import numpy as np
from prml.tensor import Parameter
from prml.nn import Network


class Affine(Network):

    def __init__(self):
        super().__init__(
            w=np.random.rand(10, 5),
            b=np.random.rand(5)
        )

    def __call__(self, x):
        return x @ self.w + self.b


class TestNetwork(unittest.TestCase):

    def test_init(self):
        affine = Affine()
        self.assertTrue("w" in affine.params)
        self.assertTrue("b" in affine.params)
        self.assertTrue(hasattr(affine, "w"))
        self.assertTrue(hasattr(affine, "b"))
        self.assertTrue(affine.w is affine.params["w"])
        self.assertTrue(affine.b is affine.params["b"])

    def test_call(self):
        x = np.random.rand(100, 10)
        affine = Affine()
        y = affine(x)
        self.assertEqual(y.shape, (100, 5))
        self.assertEqual(y.function.__class__.__name__, "Add")
        self.assertEqual(y.function.x.function.__class__.__name__, "MatMul")
        self.assertEqual(
            y.function.y.function.__class__.__name__,
            "BroadcastTo"
        )
        self.assertTrue(isinstance(y.function.y.function.x, Parameter))
        self.assertTrue(isinstance(y.function.x.function.y, Parameter))
