import unittest

from prml import nn


class TestLayer(unittest.TestCase):

    def test_layer_init(self):
        layer = nn.layers._layer._Layer(
            initializer=nn.initializers.Initializer(1.5))
        self.assertIsInstance(layer.initializer, nn.initializers.Initializer)
        self.assertEqual(layer.initializer.value, 1.5)
        self.assertIsNone(layer.bias)

    def test_layer_init_default(self):
        layer = nn.layers._layer._Layer()
        self.assertIsInstance(layer.initializer, nn.initializers.Normal)
        self.assertIsNone(layer.bias)


if __name__ == "__main__":
    unittest.main()
