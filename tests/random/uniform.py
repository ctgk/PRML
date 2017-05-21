import unittest
import numpy as np
from prml.random import Uniform


class TestUniform(unittest.TestCase):

    def test_init(self):
        u = Uniform(np.zeros(2), np.ones(2))
        self.assertTrue((u.low == np.zeros(2)).all())
        self.assertTrue((u.high == np.ones(2)).all())

    def test_repr(self):
        u = Uniform(np.zeros(2), np.ones(2))
        self.assertEqual(repr(u), "Uniform(low=[ 0.  0.], high=[ 1.  1.])")

    def test_mean(self):
        u = Uniform(-np.ones(2), np.ones(2))
        self.assertTrue((u.mean == np.zeros(2)).all())

    def test_proba(self):
        u = Uniform(-np.ones(2), np.ones(2))
        self.assertTrue(
            (u.proba(np.array([[0., 0.], [2., 0.]])) == np.array([0.25, 0.])).all()
        )

    def test_draw(self):
        u = Uniform(-np.ones(2), np.ones(2))
        sample = u.draw(1000)
        self.assertTrue(
            ((sample <= 1).all() and (sample >= -1).all())
        )