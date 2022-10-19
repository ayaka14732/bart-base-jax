from concurrent.futures import ProcessPoolExecutor
from queue import Queue
from threading import Thread

class ProcessPoolExecutorWithQueueSizeLimit(ProcessPoolExecutor):
    def __init__(self, queue_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.done_marker = object()
        self.queue_futures = Queue(maxsize=queue_size)

    def map(self, fn, *iterables):
        done_marker = self.done_marker
        queue_futures = self.queue_futures

        def usher():
            for args in zip(*iterables):
                future = self.submit(fn, *args)
                queue_futures.put(future)
            queue_futures.put(done_marker)

        thread = Thread(target=usher)
        thread.start()

        while True:
            item = queue_futures.get()
            if item is done_marker:
                break
            yield item.result()  # item is a future
