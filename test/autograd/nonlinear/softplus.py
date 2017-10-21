import unittest
import numpy as np
import prml.autograd as ag


class TestSoftplus(unittest.TestCase):

    def test_softplus(self):
        self.assertEqual(ag.softplus(0).value, np.log(2))


if __name__ == '__main__':
    unittest.main()
