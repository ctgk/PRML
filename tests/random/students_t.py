import unittest
import numpy as np
from prml.random import StudentsT


class TestStudentsT(unittest.TestCase):

    def test_init(self):
        t = StudentsT(mu=np.ones(2), precision=1, dof=1)
        self.assertTrue((t.mu == 1).all())
        self.assertTrue((t.precision == np.eye(2)).all())
        self.assertEqual(t.dof, 1)

    def test_repr(self):
        t = StudentsT(mu=np.zeros(2), precision=3, dof=2)
        self.assertEqual(
            repr(t),
            ("Student's T(\nmu=[ 0.  0.],"
             "\nprecision=\n[[ 3.  0.]\n [ 0.  3.]],\ndof=2\n)")
        )

    def test_mean(self):
        t = StudentsT(mu=np.zeros(2), precision=2, dof=2)
        self.assertTrue((t.mean == 0).all())

    def test_var(self):
        t = StudentsT(mu=np.zeros(2), precision=3, dof=3)
        self.assertTrue((t.var == np.eye(2)).all())
