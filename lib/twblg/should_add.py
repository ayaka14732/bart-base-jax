from os.path import abspath, dirname, join
import string

here = dirname(abspath(__file__))

def get_all_chars_in_data(path: str) -> set:
    s = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            _, _, mandarin, hokkien = line.rstrip('\n').split('\t')
            s.update(mandarin)
            s.update(hokkien)
    return s

should_add = get_all_chars_in_data(join(here, 'data.tsv')) | set(string.ascii_letters)
