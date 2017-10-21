import unittest
import prml.autograd as ag


class TestSigmoid(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(ag.sigmoid(0).value, 0.5)


if __name__ == '__main__':
    unittest.main()
