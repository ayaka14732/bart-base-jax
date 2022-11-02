import fileinput
from typing import Literal

def load_sentences(split=Literal['train', 'dev', 'test']) -> list[tuple[str, str]]:
    if split not in ('train', 'dev', 'test'):
        raise ValueError("`split` should be one of ('train', 'dev', 'test')")

    if split == 'train':
        with fileinput.input(files=(f'{split}.en.txt', '../abc-cantonese-parallel-corpus/en.txt'), encoding='utf-8') as f:
            en = [line.rstrip('\n') for line in f]

        with fileinput.input(files=(f'{split}.yue.txt', '../abc-cantonese-parallel-corpus/yue.txt'), encoding='utf-8') as f:
            yue = [line.rstrip('\n') for line in f]

    else:
        with open(f'{split}.en.txt', encoding='utf-8') as f:
            en = [line.rstrip('\n') for line in f]

        with open(f'{split}.yue.txt', encoding='utf-8') as f:
            yue = [line.rstrip('\n') for line in f]

    return list(zip(en, yue))
