import unittest
import numpy as np
import prml.autograd as ag


class TestDeterminant(unittest.TestCase):

    def test_determinant(self):
        A = np.array([
            [2., 1.],
            [1., 3.]
        ])
        detA = np.linalg.det(A)
        self.assertTrue((detA == ag.linalg.det(A).value).all())

        A = ag.Parameter(A)
        for _ in range(100):
            A.cleargrad()
            detA = ag.linalg.det(A)
            loss = ag.square(detA - 1)
            loss.backward()
            A.value -= 0.1 * A.grad
        self.assertAlmostEqual(detA.value, 1.)


if __name__ == '__main__':
    unittest.main()
