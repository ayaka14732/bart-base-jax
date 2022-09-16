import jax; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np
from transformers import BertTokenizer, FlaxBartModel

from lib.param_utils.save_params import save_params
from lib.random.wrapper import seed2key
from lib.twblg.CharBasedTokeniser import CharBasedTokeniser
from lib.twblg.filter_criteria import should_remove
from lib.twblg.random_init_embed import random_init_embed
from lib.twblg.should_add import should_add

tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
model = FlaxBartModel.from_pretrained('fnlp/bart-base-chinese', from_pt=True)

key = seed2key(42)

ch2id_old = {c: i for c, i in tokenizer.vocab.items() if not should_remove(c)}
new_chars = set(c for c in should_add if c not in ch2id_old)

emb_old = model.params['shared']['embedding']
emb_new = random_init_embed(key, len(new_chars))

d_merge = {
    **{c: emb_old[i] for c, i in ch2id_old.items()},
    **{c: emb_new[i] for i, c in enumerate(new_chars)},
}

vocab = []
emb_all = []

for i, (c, emb) in enumerate(d_merge.items()):
    assert isinstance(c, str)
    assert emb.shape[0] == 768
    assert c != '\n'

    c = {
        '[CLS]': '[BOS]',
        '[SEP]': '[EOS]',
        '[MASK]': '[MSK]',
    }.get(c, c)

    vocab.append(c)
    emb_all.append(emb)

emb_all = np.vstack(emb_all)

tokeniser = CharBasedTokeniser(vocab=vocab)
tokeniser.save_vocabulary()

save_params(emb_all, 'embed_params.dat')
