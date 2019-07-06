import unittest

import numpy as np

from prml import autodiff
from prml import bayesnet


class TestGaussian(unittest.TestCase):

    def test_gaussian(self):
        pass
        # qmu_m = autodiff.zeros(1)
        # qmu_s = autodiff.array([np.log(0.1)])
        # for _ in range(10000):
        #     pmu = bayesnet.Gaussian(0, 0.1)
        #     qmu = bayesnet.Gaussian(qmu_m, autodiff.exp(qmu_s))
        #     mu = qmu.sample()
        #     px = bayesnet.Gaussian(mu, 0.1)
        #     elbo = px.log_pdf([0.8]).sum() + pmu.log_pdf(mu) - qmu.log_pdf(mu)
        #     autodiff.backprop(elbo)
        #     qmu_m.value += 1e-3 * qmu_m.grad
        #     qmu_s.value += 1e-3 * qmu_s.grad
        # self.assertAlmostEqual(0.4, qmu_m.value[0], places=1)
        # self.assertAlmostEqual(
        #     np.sqrt(0.005), autodiff.exp(qmu_s).value[0], places=1)

    def test_bayesnet(self):

        class Q(bayesnet.Gaussian):

            def __init__(self):
                super().__init__(out=["mu"], name="q")
                with self.set_parameter():
                    self.mean = autodiff.zeros(1)
                    self.logstd = autodiff.array([np.log(0.1)])

            def forward(self):
                return {"mean": self.mean, "std": autodiff.exp(self.logstd)}

        class P(bayesnet.Gaussian):

            def __init__(self):
                super().__init__(out=["x"], conditions=["mu"], name="p")

            def forward(self, mu):
                return {"mean": mu, "std": 0.1}

        q = Q()
        p = P() * bayesnet.Gaussian(out=["mu"], mean=0, std=0.1)
        optimizer = autodiff.optimizer.Adam(q.parameter, 1e-2)
        for _ in range(100):
            loss = bayesnet.kl_divergence(q, p, x=[0.8])
            optimizer.minimize(loss)
        actual = q.forward()
        actual_mean = actual["mean"].value[0]
        actual_std = actual["std"].value[0]
        self.assertAlmostEqual(0.4, actual_mean, places=1)
        self.assertAlmostEqual(np.sqrt(0.005), actual_std, places=1)


if __name__ == "__main__":
    unittest.main()
