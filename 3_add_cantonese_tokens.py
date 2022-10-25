from collections import Counter
from random import Random
from transformers import BertTokenizerFast

from lib.dataset.lihkg.load_lihkg import load_lihkg
from lib.vocab import token_to_token_id as vocab_old, is_unused, is_cjkv

rng = Random(42)
sentences = load_lihkg()
sentences = rng.choices(sentences, k=524288)

tokenizer_old = BertTokenizerFast.from_pretrained('fnlp/bart-base-chinese')
# tokenizer_new = tokenizer_old.train_new_from_iterator(sentences, 2048, length=len(sentences))

########

vocab_new = set()

with open('vocab_mapping.txt', encoding='utf-8') as f:
    for line in f:
        token, token_id = line.rstrip('\n').rsplit(' ', 1)

        if is_unused(token):
            continue

        vocab_new.add(token)

########

counter = Counter()

for sentence in sentences:
    for c in sentence:
        if is_cjkv(c) and c not in vocab_new:
            counter[c] += 1

cjkv_new = set((token for token, _ in counter.most_common(150)))

########

with open('yue.txt', encoding='utf-8') as f:
    text = f.read()

for c in text:
    if is_cjkv(c) and c not in vocab_new:
        cjkv_new.add(c)

########

cs = set()
for c in cjkv_new:
    if c not in vocab_old:
        cs.append(c)

########

with open('add_token.txt', 'w', encoding='utf-8') as f:
    for c in sorted(cs):
        print(c, file=f)
