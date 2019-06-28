import unittest
import numpy as np
from prml.autodiff import config


class TestConfig(unittest.TestCase):

    def test_dtype_getter(self):
        self.assertIs(np.float32, config.dtype)
        config._dtype = np.float64
        self.assertIs(np.float64, config.dtype)

    def test_dtype_setter(self):
        config.dtype = np.float16
        self.assertIs(np.float16, config._dtype)
        config.dtype = np.float32
        self.assertIs(np.float32, config._dtype)
        config.dtype = np.float64
        self.assertIs(np.float64, config._dtype)

    def test_dtype_setter_raise(self):
        with self.assertRaises(ValueError):
            config.dtype = float
        with self.assertRaises(ValueError):
            config.dtype = int
        with self.assertRaises(ValueError):
            config.dtype = np.int

    def test_enable_backprop_getter(self):
        self.assertIs(True, config.enable_backprop)
        config._enable_backprop = False
        self.assertIs(False, config.enable_backprop)

    def test_enable_backprop_setter(self):
        config.enable_backprop = False
        self.assertIs(False, config._enable_backprop)
        config.enable_backprop = True
        self.assertIs(True, config._enable_backprop)

    def test_enable_backprop_setter_raise(self):
        with self.assertRaises(TypeError):
            config.enable_backprop = 3
        with self.assertRaises(TypeError):
            config.enable_backprop = "FALSE"


if __name__ == "__main__":
    unittest.main()
