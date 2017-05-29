import unittest
import numpy as np
from prml.random import Categorical, Dirichlet


class TestCategorical(unittest.TestCase):

    def test_init(self):
        c = Categorical()
        self.assertTrue(c.prob is None)
        c = Categorical(np.ones(3) / 3)
        self.assertEqual(c.n_classes, 3)
        self.assertTrue(np.allclose(c.prob, np.ones(3) / 3))

    def test_repr(self):
        c = Categorical(prob=np.ones(2) / 2)
        self.assertEqual(repr(c), "Categorical(prob=[ 0.5  0.5])")

    def test_mean(self):
        c = Categorical(prob=np.ones(4) / 4)
        self.assertTrue((c.mean == c.prob).all())

    def test_ml(self):
        c = Categorical()
        c.ml(np.array([[0, 1], [1, 0], [1, 0], [1, 0]]))
        self.assertEqual(c.n_classes, 2)
        self.assertTrue((c.prob == np.array([0.75, 0.25])).all())

    def test_bayes(self):
        mu = Dirichlet(concentration=np.ones(3))
        model = Categorical(prob=mu)
        self.assertEqual(
            repr(model),
            "Categorical(prob=Dirichlet(concentration=[ 1.  1.  1.]))"
        )
        model.bayes(np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))
        self.assertEqual(
            repr(model),
            "Categorical(prob=Dirichlet(concentration=[ 3.  2.  1.]))"
        )

    def test_pdf(self):
        c = Categorical(prob=np.ones(4) / 4)
        self.assertTrue((c.pdf(np.eye(4)) == 0.25).all())

    def test_draw(self):
        c = Categorical(np.ones(4) / 4)
        self.assertTrue(
            np.allclose(np.mean(c.draw(1000), axis=0), [0.25] * 4, 1e-1, 1e-1)
        )
