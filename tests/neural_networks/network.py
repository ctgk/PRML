import unittest
import numpy as np
import prml.neural_networks as pn


class Test(unittest.TestCase):

    def test_mlp(self):
        model = pn.Network()
        model.add(pn.layers.MatMul(1, 4))
        model.add(pn.layers.BiasAdd(4))
        model.add(pn.layers.Tanh())
        model.add(pn.layers.MatMul(4, 1))
        model.add(pn.layers.BiasAdd(1))
        cost_func = pn.losses.SigmoidCrossEntropy()

        x = np.ones((1, 1), np.float32) * 0.5
        t = np.ones((1, 1), np.float32) * 0.5

        eps = 0.01
        x_plus_e = x + eps
        x_minus_e = x - eps
        numerical_deriv = (
            cost_func(model.forward(x_plus_e), t)
            - cost_func(model.forward(x_minus_e), t)) / (2 * eps)

        delta = cost_func.backward(model.forward(x, True), t)
        delta = model.backward(delta)
        self.assertTrue(np.allclose(numerical_deriv, delta, atol=1e-3))

    def test_cnn(self):
        model = pn.Network()
        model.add(pn.layers.Convolution2d(1, 3, 3, pad=1))
        model.add(pn.layers.BiasAdd(3))
        model.add(pn.layers.MaxPooling2d(2, 2))
        model.add(pn.layers.Reshape((1, 2 * 2 * 3)))
        model.add(pn.layers.MatMul(12, 2))
        model.add(pn.layers.BiasAdd(2))
        cost_func = pn.losses.SoftmaxCrossEntropy()

        x = np.ones((1, 4, 4, 1), np.float32) * 0.5
        t = np.ones((1, 2), np.float32) * 0.5
        e = 0.01
        eps = np.zeros_like(x)
        eps[0, 0, 0, 0] = e
        x_plus_e = x + eps
        x_minus_e = x - eps
        numerical_deriv = (
            cost_func(model.forward(x_plus_e), t)
            - cost_func(model.forward(x_minus_e), t)) / (2 * e)
        delta = cost_func.backward(model.forward(x, True), t)
        delta = model.backward(delta)
        self.assertTrue(np.allclose(numerical_deriv, delta[0, 0, 0, 0], atol=1e-3))
