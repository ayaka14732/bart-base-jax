from os.path import abspath, dirname, join

here = dirname(abspath(__file__))

# TODO: rewrite, based on Henry, 2022

def generate_tc_set():
    左_list = '金糹言食車馬'
    右_list = '巠'
    s = set()
    with open(join(here, 'liangfen.txt')) as f:
        for line in f:
            c, p = line.rstrip('\n').split(' ')
            is_tc = any(p.startswith(x) for x in 左_list) or any(p.endswith(x) for x in 右_list)
            if is_tc:
                s.add(c)
    return s

is_tc_set = generate_tc_set()

def is_trad_ch(c):
    return c in is_tc_set
