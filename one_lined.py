import jax
import jax.nn as nn
import jax.numpy as np
import sys
import transformers
from transformers import BartTokenizer, FlaxBartForSequenceClassification

transformers.logging.set_verbosity_error()

jax.config.update('jax_platform_name', 'cpu')

jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

model = FlaxBartForSequenceClassification.from_pretrained('facebook/bart-base')

bart = model.params['model']

convert_qkv = lambda params: {'kernel': params['kernel'].reshape(768, 12, 64).transpose(1, 0, 2), 'bias': params['bias'].reshape(12, 64)}

convert_transformer_encoder = lambda params: {'self_attn': {'q_proj': convert_qkv(params['self_attn']['q_proj']), 'k_proj': convert_qkv(params['self_attn']['k_proj']), 'v_proj': convert_qkv(params['self_attn']['v_proj']), 'ff': params['self_attn']['out_proj']}, 'self_attn_layer_norm': params['self_attn_layer_norm'], 'ff0': params['fc1'], 'ff1': params['fc2'], 'final_layer_norm': params['final_layer_norm']}

convert_transformer_decoder = lambda params: {'self_attn': {'q_proj': convert_qkv(params['self_attn']['q_proj']), 'k_proj': convert_qkv(params['self_attn']['k_proj']), 'v_proj': convert_qkv(params['self_attn']['v_proj']), 'ff': params['self_attn']['out_proj']}, 'self_attn_layer_norm': params['self_attn_layer_norm'], 'cross_attn': {'q_proj': convert_qkv(params['encoder_attn']['q_proj']), 'k_proj': convert_qkv(params['encoder_attn']['k_proj']), 'v_proj': convert_qkv(params['encoder_attn']['v_proj']), 'ff': params['encoder_attn']['out_proj']}, 'cross_attn_layer_norm': params['encoder_attn_layer_norm'], 'ff0': params['fc1'], 'ff1': params['fc2'], 'final_layer_norm': params['final_layer_norm']}

params = {'embedding': {'embedding': bart['shared']['embedding']}, 'encoder_embed_positions': bart['encoder']['embed_positions']['embedding'], 'decoder_embed_positions': bart['decoder']['embed_positions']['embedding'], 'encoder_embed_layer_norm': bart['encoder']['layernorm_embedding'], 'decoder_embed_layer_norm': bart['decoder']['layernorm_embedding'], 'encoder_layers': [convert_transformer_encoder(bart['encoder']['layers'][str(i)]) for i in range(6)], 'decoder_layers': [convert_transformer_decoder(bart['decoder']['layers'][str(i)]) for i in range(6)]}

lm_head = params['embedding']['embedding'].T

fwd_layer_norm = lambda params, x: ((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)) * params['scale'] + params['bias']

fwd_embedding = lambda params, x: params['embedding'][x]

fwd_linear = lambda params, x:np.dot(x, params['kernel']) + params['bias']

def fwd_attention(params: dict, src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _, _, d_k = params['q_proj']['kernel'].shape

    t = np.einsum('bhkv,bvhm->bkhm', np.where(mask, nn.softmax(np.where(mask, np.einsum('bkhm,bvhm->bhkv', fwd_linear(params['q_proj'], dst), fwd_linear(params['k_proj'], src)) / np.sqrt(d_k), np.NINF)), 0), fwd_linear(params['v_proj'], src))

    d0, d1, d2, d3 = t.shape

    t = t.reshape(d0, d1, d2 * d3)

    return fwd_linear(params['ff'], t)

def fwd_transformer_encoder(params: dict, src: np.ndarray, mask_enc: np.ndarray) -> np.ndarray:
    t = fwd_layer_norm(params['self_attn_layer_norm'], fwd_attention(params['self_attn'], src, src, mask_enc) + src)

    return fwd_layer_norm(params['final_layer_norm'], fwd_linear(params['ff1'], nn.gelu(fwd_linear(params['ff0'], t))) + t)

def fwd_transformer_decoder(params: dict, src: np.ndarray, dst: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray) -> np.ndarray:
    dst = fwd_layer_norm(params['self_attn_layer_norm'], fwd_attention(params['self_attn'], dst, dst, mask_dec) + dst)

    t = fwd_layer_norm(params['cross_attn_layer_norm'], fwd_attention(params['cross_attn'], src, dst, mask_dec_enc) + dst)

    return fwd_layer_norm(params['final_layer_norm'], fwd_linear(params['ff1'], nn.gelu(fwd_linear(params['ff0'], t))) + t)

def fwd_transformer(params: dict, src: np.ndarray, dst: np.ndarray, mask_enc: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray) -> np.ndarray:
    _, width_enc = src.shape
    _, width_dec = dst.shape

    src = fwd_layer_norm(params['encoder_embed_layer_norm'], fwd_embedding(params['embedding'], src) + params['encoder_embed_positions'][2:width_enc+2])
    for encoder_layer in params['encoder_layers']:
        src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    dst = fwd_layer_norm(params['decoder_embed_layer_norm'], fwd_embedding(params['embedding'], dst) + params['decoder_embed_positions'][2:width_dec+2])
    for decoder_layer in params['decoder_layers']:
        dst = fwd_transformer_decoder(decoder_layer, src, dst, mask_dec, mask_dec_enc)

    return dst

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

sentences = [sys.argv[1]]

batch = tokenizer(sentences, return_tensors='jax')

src = batch.input_ids

mask_enc_1d = batch.attention_mask.astype(np.bool_)

i = 1
dst = np.zeros((len(sentences), 1), dtype=np.int32)

while True:
    mask_dec_1d = np.ones((len(sentences), i), dtype=np.bool_)

    mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
    mask_dec = np.tril(np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
    mask_dec_enc = np.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]

    y = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)

    a = nn.softmax(y @ lm_head)
    a = np.argmax(a[:, -1], axis=-1)

    i += 1
    dst = np.hstack((dst, a[..., None]))
    dst

    if np.all(a == 2):
        break

print(tokenizer.batch_decode(dst, skip_special_tokens=True)[0])
