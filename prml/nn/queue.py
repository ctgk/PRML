class BackPropQueue(object):

    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def enqueue(self, array):
        array.is_in_queue = True
        self.queue.append(array)

    def dequeue(self, depth_to_dequeue):
        queue = self.queue[0]
        for candidate in self.queue:
            if candidate.depth == depth_to_dequeue:
                queue = candidate
                break
            elif candidate.depth > queue.depth:
                queue = candidate
        self.queue.remove(queue)
        queue.is_in_queue = False
        return queue


backprop_queue = BackPropQueue()

