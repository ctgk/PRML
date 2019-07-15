import unittest

import numpy as np

from prml import autodiff, nn


class TestNN(unittest.TestCase):

    def test_nn(self):
        x_train = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        y_train = np.array([0, 1, 1, 0]).reshape(4, 1)
        model = nn.Network(
            nn.layers.Dense(2, 3),
            nn.layers.Tanh(),
            nn.layers.Dense(3, 1)
        )
        optimizer = autodiff.optimizer.Adam(model.parameter, 1e-1)
        for _ in range(100):
            likelihood = autodiff.random.bernoulli_logpdf(
                y_train, model(x_train)).sum()
            optimizer.maximize(likelihood)
        output = autodiff.sigmoid(model(x_train)).value
        self.assertTrue(np.allclose(y_train, output, atol=0.1))


if __name__ == "__main__":
    unittest.main()
