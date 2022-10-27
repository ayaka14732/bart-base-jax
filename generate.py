import jax; jax.config.update('jax_platforms', 'cpu')
import jax.numpy as np
import regex as re
from transformers import BartConfig, BertTokenizer

from lib.Generator import Generator
from lib.model import fwd_embedding, fwd_layer_norm, fwd_transformer_encoder
from lib.param_utils.load_params import load_params

def fwd_encode(params: dict, src: np.ndarray, mask_enc: np.ndarray) -> np.ndarray:
    # params
    embedding: dict = params['embedding']  # embedding
    encoder_embed_positions: np.ndarray = params['encoder_embed_positions']  # array
    encoder_embed_layer_norm: dict = params['encoder_embed_layer_norm']  # layer norm
    encoder_layers: list = params['encoder_layers']  # list of transformer encoder

    _, width_enc = src.shape

    offset = 2

    # encoder
    src = fwd_embedding(embedding, src)
    src = src + encoder_embed_positions[offset:width_enc+offset]
    src = fwd_layer_norm(encoder_embed_layer_norm, src)

    for encoder_layer in encoder_layers:
        src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    return src

tokenizer = BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

params = load_params('electric-glade-5-7-40960.dat')
params = jax.tree_map(np.asarray, params)

config = BartConfig.from_pretrained('fnlp/bart-base-chinese', vocab_size=12660)
generator = Generator(params, config=config)

def clean_up_spaces(s: str) -> str:
    s = re.sub(r'(?<=[\p{Unified_Ideograph}\u3006\u3007，。！？《》]) (?=[\p{Unified_Ideograph}\u3006\u3007，。！？《》])', '', s)
    s = re.sub(r'(?<=[a-zA-Z]) (?=[.,])', '', s)
    return s

def text_infilling(sentences: list[str]) -> list[str]:
    batch = tokenizer(sentences, padding=True, return_tensors='jax')

    src = batch.input_ids.astype(np.uint16)
    mask_enc_1d = batch.attention_mask.astype(np.bool_)
    mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

    encoder_last_hidden_output = fwd_encode(params, src, mask_enc)
    generate_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=5, max_length=100)

    decoded_sentences = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decoded_sentences = [clean_up_spaces(s) for s in decoded_sentences]
    return decoded_sentences

sentences = [
    '呢兩樣嘢二選一，你會[MASK]個？',
    '企出來面對呢件事嘅人係佢，[MASK]我',
    '杯奶茶要熱定[MASK]？',
    '聽日就要返香港，我激動到[MASK]唔着',
    '記得揀有防滑功能嘅鞋底，[MASK]親呀！',
]
for result in text_infilling(sentences):
    print(result)
