import unittest
import numpy as np
import prml.autograd as ag


class TestCholesky(unittest.TestCase):

    def test_cholesky(self):
        A = np.array([
            [2., -1],
            [-1., 5.]
        ])
        L = np.linalg.cholesky(A)
        Ap = ag.Parameter(A)
        L_test = ag.linalg.cholesky(Ap)
        self.assertTrue((L == L_test.value).all())

        T = np.array([
            [1., 0.],
            [-1., 2.]
        ])
        for _ in range(1000):
            Ap.cleargrad()
            L_ = ag.linalg.cholesky(Ap)
            loss = ag.square(T - L_).sum()
            loss.backward()
            Ap.value -= 0.1 * Ap.grad

        self.assertTrue(np.allclose(Ap.value, T @ T.T))


if __name__ == '__main__':
    unittest.main()
