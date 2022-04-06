from os.path import expanduser
from random import choices

zh = expanduser('~/dataset/processed/wikimatrix21.zh')
en = expanduser('~/dataset/processed/wikimatrix21.en')

with open(zh) as fzh, open(en) as fen:
    pairs = list(zip(fzh, fen))

for a, b in choices(pairs, k=5):
    print(f'{a}{b}')
