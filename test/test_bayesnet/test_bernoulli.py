import unittest

import numpy as np

from prml import autodiff
from prml import bayesnet


class TestBernoulli(unittest.TestCase):

    def test_bayes(self):

        class Posterior(bayesnet.Beta):
            def __init__(self):
                super().__init__(out=["mu"], name="q")
                with self.set_parameter():
                    self.a = autodiff.ones(1)
                    self.b = autodiff.ones(1)

            def forward(self):
                return {"a": self.a, "b": self.b}

        class Model(bayesnet.Bernoulli):
            def __init__(self):
                super().__init__(out=["x"], conditions=["mu"], name="p")

            def forward(self, mu):
                return {"mean": mu}

        q = Posterior()
        p = Model() * bayesnet.Beta(out=["mu"], a=1, b=1)
        optimizer = autodiff.optimizer.Momentum(q.parameter, 1e-2)
        for _ in range(5000):
            loss = bayesnet.kl_divergence(q, p, x=np.array([1, 1, 0]))
            optimizer.minimize(loss)
            actual = q.forward()
            actual_a = actual["a"].value[0]
            actual_b = actual["b"].value[0]
            print(loss.value[0], actual_a, actual_b)


if __name__ == "__main__":
    unittest.main()
