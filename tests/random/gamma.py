import unittest
import numpy as np
from prml.random import Gamma


class TestGamma(unittest.TestCase):

    def test_init(self):
        g = Gamma(shape=4, rate=6)
        self.assertEqual(g.shape, 4)
        self.assertEqual(g.rate, 6)

    def test_repr(self):
        g = Gamma(shape=1, rate=1)
        self.assertEqual(repr(g), "Gamma(shape=1, rate=1)")

    def test_mul(self):
        g = Gamma(shape=1, rate=2)
        h = g * 2
        self.assertEqual(repr(g), "Gamma(shape=1, rate=2)")
        self.assertEqual(repr(h), "Gamma(shape=1, rate=1.0)")
        h = 2 * h
        self.assertEqual(repr(h), "Gamma(shape=1, rate=0.5)")
        h *= 0.5
        self.assertEqual(repr(h), "Gamma(shape=1, rate=1.0)")

    def test_div(self):
        g = Gamma(shape=1, rate=2)
        h = g / 2
        self.assertEqual(repr(g), "Gamma(shape=1, rate=2)")
        self.assertEqual(repr(h), "Gamma(shape=1, rate=4)")
        h /= 2
        self.assertEqual(repr(h), "Gamma(shape=1, rate=8)")

    def test_mean(self):
        g = Gamma(shape=4, rate=6)
        self.assertEqual(g.mean, 2 / 3)

    def test_var(self):
        g = Gamma(shape=4, rate=6)
        self.assertEqual(g.var, 1 / 9)

    def test_pdf(self):
        g = Gamma(shape=4, rate=6)
        self.assertTrue(np.allclose(g.pdf(np.ones((1, 1))), 0.53541047))

    def test_draw(self):
        g = Gamma(shape=4, rate=6)
        self.assertTrue(np.allclose(g.draw(10000).mean(axis=0), 2 / 3, 1e-1, 1e-1))
