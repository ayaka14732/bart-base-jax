from glob import glob
from multiprocessing import Pool, Process, Queue
import jax.numpy as np
import os
from typing import List

def _transform(sentences: List[str]) -> np.ndarray:
    # TODO: handle batches
    return tokenizer(sentences)

def _producer(queue: Queue, n_workers: int):
    # load dataset

    all_sentences = []

    for filename in glob('dump2/*/*')[:1]:  # TODO: remove [:1]
        with open(filename, encoding='utf-8') as f:
            for line in f:
                all_sentences.append(line.rstrip('\n'))

    n_sents = len(all_sentences)

    # TODO: split dataset into batches before tokenization
    # or can share the tokenizer object?

    queue.put(n_sents)

    with Pool(processes=n_workers) as p:
        for sentence in p.imap(_transform, all_sentences):
            queue.put(sentence)

def on_demand_dataloader(n_workers: int=None):
    if n_workers is None:
        n_workers = os.cpu_count()

    queue = Queue(maxsize=n_workers)

    producer_proc = Process(target=_producer, args=(queue, n_workers))
    producer_proc.start()

    length = queue.get()

    return (queue.get() for _ in range(length))
