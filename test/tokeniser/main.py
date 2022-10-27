import jax; jax.config.update('jax_platforms', 'cpu')
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lib.tokeniser import BartTokenizerWithoutOverflowEOS

tokenizer = BartTokenizerWithoutOverflowEOS.from_pretrained('Ayaka/bart-base-cantonese')

sentences = ['a a', 'go go go', 'hi hi hi hi', '㷫㷫㷫㷫㷫']
max_length = 5
input_ids, attention_masks = tokenizer(sentences, max_length)

assert input_ids.tolist() == \
    [[101, 143, 143, 102, 0],
    [101, 6956, 6956, 6956, 102],
    [101, 7496, 7496, 7496, 7496],
    [101, 12, 12, 12, 12]]
assert attention_masks.tolist() == \
    [[True, True, True, True, False],
    [True, True, True, True, True],
    [True, True, True, True, True],
    [True, True, True, True, True]]
