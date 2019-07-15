import unittest
from unittest.mock import MagicMock

import numpy as np

from prml import autodiff
from prml.autodiff._core._backprop import _BackPropTaskManager


class TestBackPropTaskManager(unittest.TestCase):

    def setUp(self):
        self.bptm = _BackPropTaskManager(clear_previous_grad=False)

    def test_init(self):
        self.assertSetEqual(set(), self.bptm._tasks)

    def test_add_task(self):
        a = autodiff.array(1)
        self.bptm.add_task(a)
        self.assertSetEqual(set([a]), self.bptm._tasks)

    def test_add_task_raise(self):
        self.assertRaises(TypeError, self.bptm.add_task, variable="abc")
        self.assertSetEqual(set(), self.bptm._tasks)

    def test_len(self):
        self.assertEqual(0, len(self.bptm))
        for i in range(1, 7):
            self.bptm.add_task(autodiff.array(i))
            self.assertEqual(i, len(self.bptm))

    def test_in(self):
        a = autodiff.array(1)
        self.bptm.add_task(a)
        self.assertIn(a, self.bptm)

    def test_not_in(self):
        a = autodiff.array(1)
        b = autodiff.array(1)
        self.bptm.add_task(a)
        self.assertNotIn(b, self.bptm)

    def test_get_next_task(self):
        mock_tasks = set([MagicMock(_depth=i) for i in range(10)])
        self.bptm._tasks = mock_tasks
        for i in range(9, -1, -1):
            task = self.bptm.pop_next_task()
            self.assertEqual(i, task._depth)


class TestBackprop(unittest.TestCase):

    def test_backprop(self):
        def sigmoid(x):
            return np.tanh(x * 0.5) * 0.5 + 0.5

        x_train = autodiff.asarray(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        y_train = autodiff.asarray(np.array([[0], [1], [1], [0]]))
        w1 = autodiff.asarray(np.random.normal(0, 1, (2, 3)))
        b1 = autodiff.asarray(np.zeros(3))
        w2 = autodiff.asarray(np.random.normal(0, 1, (3, 1)))
        b2 = autodiff.asarray(np.zeros(1))

        h = x_train @ w1 + b1
        a = autodiff.tanh(h)
        logit = a @ w2 + b2
        loss = autodiff.sigmoid_cross_entropy(y_train, logit)
        autodiff.backprop(loss)
        dlogit = sigmoid(logit.value) - y_train.value
        db2 = np.sum(dlogit, axis=0)
        dw2 = a.value.T @ dlogit
        da = dlogit @ w2.value.T
        dh = da * (1 - a.value ** 2)
        db1 = np.sum(dh, axis=0)
        dw1 = x_train.value.T @ dh
        dx = dh @ w1.value.T
        self.assertTrue(np.allclose(logit.grad, dlogit))
        self.assertTrue(np.allclose(b2.grad, db2))
        self.assertTrue(np.allclose(w2.grad, dw2))
        self.assertTrue(np.allclose(a.grad, da))
        self.assertTrue(np.allclose(h.grad, dh))
        self.assertTrue(np.allclose(b1.grad, db1))
        self.assertTrue(np.allclose(w1.grad, dw1))
        self.assertTrue(np.allclose(x_train.grad, dx))


if __name__ == "__main__":
    unittest.main()
