import numpy as np

from prml.autodiff._core._array import Array
from prml.autodiff._core._config import config


class _BackPropTaskManager(object):

    def __init__(self, clear_previous_grad: bool):
        self.clear_previous_grad = clear_previous_grad
        self._tasks = set()

    def __len__(self):
        return len(self._tasks)

    def __contains__(self, task: Array):
        return task in self._tasks

    def add_task(self, task: Array):
        if not isinstance(task, Array):
            raise TypeError
        if self.clear_previous_grad and task not in self:
            task.cleargrad()
        self._tasks.add(task)

    def pop_next_task(self):
        task = max(self._tasks, key=lambda x: x._depth)
        self._tasks.discard(task)
        return task


def backprop(array: Array, grad=None, clear_previous_grad: bool = True):
    if grad is None:
        grad = np.ones_like(array.value).astype(config.dtype)
    assert(grad.shape == array.value.shape)
    backprop_taskmanager = _BackPropTaskManager(clear_previous_grad)
    backprop_taskmanager.add_task(array)
    array._accumulate_gradient_from_child(grad)
    while len(backprop_taskmanager):
        task = backprop_taskmanager.pop_next_task()
        if task._parent is not None:
            task._parent.backward(task._gradtmp, backprop_taskmanager)
        task.update_grad(task._gradtmp)
        task.gradtmp = None
