import unittest
from unittest.mock import MagicMock

from prml.autodiff import array
from prml.autodiff.core.backward import BackPropTaskManager


class TestBackPropTaskManager(unittest.TestCase):

    def setUp(self):
        self.bptm = BackPropTaskManager()

    def test_init(self):
        self.assertSetEqual(set(), self.bptm._tasks)

    def test_add_task(self):
        a = array(1)
        self.bptm.add_task(a)
        self.assertSetEqual(set([a]), self.bptm._tasks)

    def test_add_task_raise(self):
        self.assertRaises(TypeError, self.bptm.add_task, variable="abc")
        self.assertSetEqual(set(), self.bptm._tasks)

    def test_len(self):
        self.assertEqual(0, len(self.bptm))
        for i in range(1, 7):
            self.bptm.add_task(array(i))
            self.assertEqual(i, len(self.bptm))

    def test_in(self):
        a = array(1)
        self.bptm.add_task(a)
        self.assertIn(a, self.bptm)

    def test_not_in(self):
        a = array(1)
        b = array(1)
        self.bptm.add_task(a)
        self.assertNotIn(b, self.bptm)

    def test_get_next_task(self):
        mock_tasks = set([MagicMock(_depth=i) for i in range(10)])
        self.bptm._tasks = mock_tasks
        for i in range(9, -1, -1):
            task = self.bptm.pop_next_task()
            self.assertEqual(i, task._depth)


if __name__ == "__main__":
    unittest.main()
