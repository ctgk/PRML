import unittest
import numpy as np
import prml.autograd as ag


class TestLogdet(unittest.TestCase):

    def test_logdet(self):
        A = np.array([
            [2., 1.],
            [1., 3.]
        ])
        logdetA = np.linalg.slogdet(A)[1]
        self.assertTrue((logdetA == ag.linalg.logdet(A).value).all())

        A = ag.Parameter(A)
        for _ in range(100):
            A.cleargrad()
            logdetA = ag.linalg.logdet(A)
            loss = ag.square(logdetA - 1)
            loss.backward()
            A.value -= 0.1 * A.grad
        self.assertAlmostEqual(logdetA.value, 1)


if __name__ == '__main__':
    unittest.main()
