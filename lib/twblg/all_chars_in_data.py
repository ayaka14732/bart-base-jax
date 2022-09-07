from os.path import abspath, dirname, join

here = dirname(abspath(__file__))

def get_all_chars_in_data(path: str) -> set:
    s = set()
    with open(path, encoding='utf-8') as f:
        s = set(f.read())
    s.remove('\t')
    s.remove('\n')
    return s

all_chars_in_data = get_all_chars_in_data(join(here, 'data.tsv'))

def dump_all_chars_in_data(path: str, dst: str) -> None:
    with open(dst, encoding='utf-8') as f:
        for c in get_all_chars_in_data(path):
            print(c, file=f)
