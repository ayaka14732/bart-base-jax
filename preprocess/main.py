import csv
from itertools import zip_longest
import random
import regex as re
from StarCC import PresetConversion

convert_tw2cn = PresetConversion(src='tw', dst='cn', with_phrase=False)

def segmentation(例句, 例句標音):
    '''
    ```python
    >>> 例句 = '塗跤一半擺仔無拭，袂偌垃圾啦！'
    >>> 例句標音 = 'Thôo-kha tsi̍t-puànn-pái-á bô tshit, bē guā lah-sap--lah!'
    >>> segmentation(例句, 例句標音)
    '塗跤|一半擺仔|無|拭|，|袂|偌|垃圾啦|！'
    ```
    '''
    s = 例句標音.replace('--', '-')
    parts = re.split(r' |(?=-)|(?=\p{Punct})', s)
    res = []
    for ch, part in zip_longest(例句, parts):
        assert ch is not None, (例句, 例句標音)
        assert part is not None, (例句, 例句標音)
        if not part.startswith('-'):
            res.append('/')
        res.append(ch)
    return ''.join(res[1:])

output = []

with open('例句-1.csv', encoding='utf-8', newline='') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for line in reader:
        try:
            _, _, _, _, 例句, 例句標音, 華語翻譯 = line

            if len(例句) <= 5 or len(華語翻譯) <= 5:
                continue

            if abs(len(例句) - len(華語翻譯)) >= 5:  # 含有解說
                continue

            例句標音 = 例句標音.rstrip()
            例句 = segmentation(例句, 例句標音)
            output.append((華語翻譯, 例句))
        except Exception:
            pass

random.seed(42)
random.shuffle(output)

d = {}

for 華語, 台語 in output:
    华语 = convert_tw2cn(華語)
    台语 = convert_tw2cn(台語)
    d[華語, 台語, 华语, 台语] = None

with open('data.tsv', 'w', encoding='utf-8') as f:
    for 華語, 台語, 华语, 台语 in d:
        print(華語, 台語, 华语, 台语, sep='\t', file=f)
