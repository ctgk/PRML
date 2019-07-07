import unittest

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from prml import autodiff, nn


class TestCNN(unittest.TestCase):

    def test_cnn(self):
        x, y = load_digits(return_X_y=True)
        x = x / 16.
        x = x.reshape(-1, 8, 8, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, stratify=y)
        model = nn.Sequential(
            nn.layers.Convolution2d(1, 20, 3, 1, 0, autodiff.relu),
            nn.layers.MaxPooling2d(2, 2),
            nn.layers.Convolution2d(20, 40, 3, 1, 0, autodiff.relu),
            nn.layers.Flatten(),
            nn.layers.Dense(40, 10)
        )
        optimizer = autodiff.optimizer.Adam(model.parameter, 0.1)
        for i in range(20):
            logit = model(x_train)
            loss = autodiff.random.softmax_cross_entropy(y_train, logit).mean()
            optimizer.minimize(loss)
        self.assertLess(loss.value[0], 0.5)


if __name__ == "__main__":
    unittest.main()
