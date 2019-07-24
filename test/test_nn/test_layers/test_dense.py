import unittest

import numpy as np

from prml import autodiff, nn


class TestDense(unittest.TestCase):

    def test_dense_init_default(self):
        dense = nn.layers.Dense(2, 3)
        self.assertIsInstance(dense.initializer, nn.initializers.Normal)
        self.assertIsInstance(dense.weight, autodiff.Array)
        self.assertTrue(dense.weight.requires_grad)
        self.assertTupleEqual(dense.weight.shape, (2, 3))
        self.assertIsInstance(dense.bias, autodiff.Array)
        self.assertEqual(dense.bias.size, 3)
        self.assertTrue(dense.bias.requires_grad)
        self.assertDictEqual(
            dense.parameter,
            {"Dense.weight": dense.weight, "Dense.bias": dense.bias})

    def test_dense_init(self):
        dense = nn.layers.Dense(
            4, 5, initializer=nn.initializers.Initializer(2),
            has_bias=False)
        self.assertIsInstance(dense.weight, autodiff.Array)
        self.assertTupleEqual(dense.weight.shape, (4, 5))
        self.assertTrue(np.allclose(2, dense.weight.value))
        self.assertIsNone(dense.bias)
        self.assertDictEqual(dense.parameter, {"Dense.weight": dense.weight})

    def test_forward(self):
        dense = nn.layers.Dense(7, 3)
        x = np.random.randn(10, 7)

        expect = x @ dense.weight.value + dense.bias.value
        actual = dense.forward(x).value
        self.assertTrue(np.allclose(expect, actual))

    def test_forward_no_bias(self):
        dense = nn.layers.Dense(7, 3, has_bias=False)
        x = np.random.randn(10, 7)

        expect = x @ dense.weight.value
        actual = dense.forward(x).value
        self.assertTrue(np.allclose(expect, actual))


if __name__ == "__main__":
    unittest.main()
