import unittest

import numpy as np

from prml import autodiff
from prml import bayesnet


class TestBernoulli(unittest.TestCase):

    def test_bayes(self):

        class Posterior(bayesnet.functions.Beta):
            def __init__(self):
                super().__init__(var=["mu"], name="q")
                with self.initialize():
                    self.a = autodiff.ones(1)
                    self.b = autodiff.ones(1)

            def forward(self):
                return {"a": self.a, "b": self.b}

        class Model(bayesnet.functions.Bernoulli):
            def __init__(self):
                super().__init__(var=["x"], conditions=["mu"], name="p")

            def forward(self, mu):
                return {"mean": mu}

        q = Posterior()
        p = Model() * bayesnet.functions.Beta(var=["mu"], a=1, b=1)
        optimizer = autodiff.optimizer.Momentum(q.parameter, 1e-2)
        for _ in range(5000):
            loss = bayesnet.kl_divergence(q, p, x=np.array([1, 1, 0]))
            optimizer.minimize(loss)
        actual = q.forward()
        actual_a = actual["a"].value[0]
        actual_b = actual["b"].value[0]
        self.assertAlmostEqual(3, actual_a, places=0)
        self.assertAlmostEqual(2, actual_b, places=0)


if __name__ == "__main__":
    unittest.main()
