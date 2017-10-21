import unittest
import prml.autograd as ag


class TestTanh(unittest.TestCase):

    def test_tanh(self):
        self.assertEqual(ag.tanh(0).value, 0)


if __name__ == '__main__':
    unittest.main()
