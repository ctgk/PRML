import unittest
import numpy as np
import prml.nn as nn


class TestBackward(unittest.TestCase):

    def test_backward(self):
        def sigmoid(x):
            return np.tanh(x * 0.5) * 0.5 + 0.5

        x_train = nn.asarray(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        y_train = nn.asarray(np.array([[0], [1], [1], [0]]))
        w1 = nn.asarray(np.random.normal(0, 1, (2, 3)))
        b1 = nn.asarray(np.zeros(3))
        w2 = nn.asarray(np.random.normal(0, 1, (3, 1)))
        b2 = nn.asarray(np.zeros(1))
        parameter = [w1, b1, w2, b2]
        for _ in range(10):
            for param in parameter:
                param.cleargrad()
            x_train.cleargrad()
            h = x_train @ w1 + b1
            a = nn.tanh(h)
            logit = a @ w2 + b2
            loss = nn.loss.sigmoid_cross_entropy(logit, y_train)
            loss.backward()
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
