import unittest
import numpy as np
import prml.autograd as ag


class TestSolve(unittest.TestCase):

    def test_solve(self):
        A = np.array([
            [2., 1.],
            [1., 3.]
        ])
        B = np.array([1., 2.])[:, None]
        AinvB = np.linalg.solve(A, B)
        self.assertTrue((AinvB == ag.linalg.solve(A, B).value).all())

        A = ag.Parameter(A)
        B = ag.Parameter(B)
        for _ in range(100):
            A.cleargrad()
            B.cleargrad()
            AinvB = ag.linalg.solve(A, B)
            loss = ag.square(AinvB - 1).sum()
            loss.backward()
            A.value -= A.grad
            B.value -= B.grad
        self.assertTrue(np.allclose(AinvB.value, 1))


if __name__ == '__main__':
    unittest.main()
