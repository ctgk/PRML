import unittest
import numpy as np
import prml.nn as nn


class TestGaussian(unittest.TestCase):

    def test_gaussian_draw_forward(self):
        mu = nn.array(0)
        sigma = nn.softplus(nn.array(-1))
        gaussian = nn.Gaussian(mu, sigma)
        sample = []
        for _ in range(1000):
            sample.append(gaussian.draw().value)
        self.assertTrue(np.allclose(np.mean(sample), 0, rtol=0.1, atol=0.1), np.mean(sample))
        self.assertTrue(np.allclose(np.std(sample), gaussian.std.value, 0.1, 0.1))

    def test_gaussian_draw_backward(self):
        mu = nn.array(0)
        s = nn.array(2)
        optimizer = nn.optimizer.Gradient({0: mu, 1: s}, 0.01)
        prior = nn.Gaussian(1, 1)
        for _ in range(1000):
            mu.cleargrad()
            s.cleargrad()
            gaussian = nn.Gaussian(mu, nn.softplus(s))
            gaussian.draw()
            loss = nn.loss.kl_divergence(gaussian, prior).sum()
            optimizer.minimize(loss)
        self.assertTrue(np.allclose(gaussian.mean.value, 1, 0.1, 0.1))
        self.assertTrue(np.allclose(gaussian.std.value, 1, 0.1, 0.1))


if __name__ == "__main__":
    unittest.main()
