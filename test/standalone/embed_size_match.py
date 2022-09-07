import jax; jax.config.update('jax_platforms', 'cpu')
from transformers import BertTokenizer, FlaxBartModel

tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
model = FlaxBartModel.from_pretrained('fnlp/bart-base-chinese', from_pt=True)

vocab_size = tokenizer.vocab_size  # 21128
embed_size = 768

embed_param_shape = model.params['shared']['embedding'].shape
assert embed_param_shape == (vocab_size, embed_size)
