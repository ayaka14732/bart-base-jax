import jax
import jax.numpy as np
import sys
from transformers import BartTokenizer, BertTokenizer
from os.path import expanduser

from lib.param_utils.load_params import load_params
from lib.fwd_embedding import fwd_embedding
from lib.fwd_layer_norm import fwd_layer_norm
from lib.fwd_transformer_encoder import fwd_transformer_encoder
from lib.generator import Generator

jax.config.update('jax_platform_name', 'cpu')

assert len(sys.argv) == 2, 'Please provide the path to model checkpoint (*.dat) as a command line argument'

def fwd_encode(params: dict, src: np.ndarray, mask_enc: np.ndarray) -> np.ndarray:
    # pparams_ch
    params_ch = params['ch']
    ch_embedding: dict = params_ch['embedding']  # embedding
    ch_encoder_embed_positions: np.ndarray = params_ch['encoder_embed_positions']  # array
    ch_encoder_embed_layer_norm: dict = params_ch['encoder_embed_layer_norm']  # layer norm
    ch_encoder_layers: list = params_ch['encoder_layers']  # list of transformer encoder

    # params
    encoder_layers: list = params['encoder_layers']  # list of transformer encoder

    _, width_enc = src.shape

    offset = 2

    # ch encoder
    src = fwd_embedding(ch_embedding, src)
    src = src + ch_encoder_embed_positions[offset:width_enc + offset]
    src = fwd_layer_norm(ch_encoder_embed_layer_norm, src)
    for encoder_layer in ch_encoder_layers:
        src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    # encoder
    for encoder_layer in encoder_layers:
        src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    return src

tokenizer_zh = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
tokenizer_en = BartTokenizer.from_pretrained('facebook/bart-base')

params = load_params(sys.argv[1])
params = jax.tree_map(np.asarray, params)
generator = Generator(params)

sentences = [
    '毕业之后，我的兴趣是在游泳池驾驶飞机。',
    '果然步要晚上出来跑，大下午的跑就是找暑中。',
    '对，而且这航班还得放包在座椅底下，腿更没空间活动。',
    '整个出入境过程只用过登机牌，护照，居留卡都没人看。',
    '全波兰的报纸和杂志中约有半数是在华沙印刷的。',
    '这个数量只足以运行一间塔吉克发电厂。',
]

# with open(expanduser('~/dataset/processed/Eval/sources/newstest2021.zh-en.src.zh')) as f:
#     for line in f:
#         line = line.rstrip('\n')
#         sentences.append(line)

batch = tokenizer_zh(sentences, padding=True, return_tensors='jax')

src = batch.input_ids
mask_enc_1d = batch.attention_mask.astype(np.bool_)
mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

encoder_last_hidden_output = fwd_encode(params, src, mask_enc)
generate_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=5, max_length=100000)
decoded_sentences = tokenizer_en.batch_decode(generate_ids, skip_special_tokens=True)

for translated_sentence in decoded_sentences:
    print(translated_sentence)

# with open('test_output2.txt', 'w') as f:
#     for translated_sentence in decoded_sentences:
#         print(translated_sentence)
#         print(translated_sentence, file=f)
