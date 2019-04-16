import unittest
import numpy as np
import prml.nn as nn


class TestBernoulli(unittest.TestCase):

    def test_bernoulli_kl(self):
        logit = nn.random.normal(0, 1, (5, 3))
        mu = nn.random.uniform(0, 1, (5, 3))
        q = nn.Bernoulli(mu)
        for _ in range(1000):
            logit.cleargrad()
            p = nn.Bernoulli(logit=logit)
            nn.loss.kl_divergence(p, q).backward()
            logit.value -= 0.1 * logit.grad
        self.assertTrue(np.allclose(p.mean.value, q.mean.value, 1e-2, 1e-2))
        self.assertTrue(logit.value.dtype, nn.config.dtype)
        self.assertEqual(p.mean.value.dtype, nn.config.dtype)
        self.assertEqual(q.mean.value.dtype, nn.config.dtype)
        self.assertEqual(nn.config.dtype, np.float32)

    def test_bernoulli_mu(self):
        npmu = np.random.uniform(0, 1, (5, 3))
        npt = np.random.uniform(0, 1, (5, 3))
        p = nn.Bernoulli(npmu)
        log_likelihood = p.log_pdf(npt)
        self.assertTrue(np.allclose(log_likelihood.value, npt * np.log(npmu) + (1 - npt) * np.log(1 - npmu)))

    def test_bernoulli_logit(self):
        nplogit = np.random.randn(5, 3)
        npmu = np.tanh(nplogit * 0.5) * 0.5 + 0.5
        npt = np.random.uniform(0, 1, (5, 3))
        p = nn.Bernoulli(logit=nplogit)
        log_likelihood = p.log_pdf(npt)
        self.assertTrue(np.allclose(log_likelihood.value, npt * np.log(npmu) + (1 - npt) * np.log(1 - npmu)))


if __name__ == "__main__":
    unittest.main()
