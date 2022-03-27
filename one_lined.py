import jax
import jax.nn as nn
import jax.numpy as np
from transformers import BartTokenizer, FlaxBartForSequenceClassification

# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

# load params

model = FlaxBartForSequenceClassification.from_pretrained('facebook/bart-base')
bart = model.params['model']

def convert_qkv(params):
    return {
        'kernel': params['kernel'].reshape(768, 12, 64).transpose(1, 0, 2),
        'bias': params['bias'].reshape(12, 64),
    }

def convert_transformer_encoder(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv(params['self_attn']['q_proj']),
            'k_proj': convert_qkv(params['self_attn']['k_proj']),
            'v_proj': convert_qkv(params['self_attn']['v_proj']),
            'ff': params['self_attn']['out_proj'],
        },
        'self_attn_layer_norm': params['self_attn_layer_norm'],
        'ff0': params['fc1'],
        'ff1': params['fc2'],
        'final_layer_norm': params['final_layer_norm'],
    }

def convert_transformer_decoder(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv(params['self_attn']['q_proj']),
            'k_proj': convert_qkv(params['self_attn']['k_proj']),
            'v_proj': convert_qkv(params['self_attn']['v_proj']),
            'ff': params['self_attn']['out_proj'],
        },
        'self_attn_layer_norm': params['self_attn_layer_norm'],
        'cross_attn': {
            'q_proj': convert_qkv(params['encoder_attn']['q_proj']),
            'k_proj': convert_qkv(params['encoder_attn']['k_proj']),
            'v_proj': convert_qkv(params['encoder_attn']['v_proj']),
            'ff': params['encoder_attn']['out_proj'],
        },
        'cross_attn_layer_norm': params['encoder_attn_layer_norm'],
        'ff0': params['fc1'],
        'ff1': params['fc2'],
        'final_layer_norm': params['final_layer_norm'],
    }

params = {
    'embedding': {'embedding': bart['shared']['embedding']},
    'encoder_embed_positions': bart['encoder']['embed_positions']['embedding'],
    'decoder_embed_positions': bart['decoder']['embed_positions']['embedding'],
    'encoder_embed_layer_norm': bart['encoder']['layernorm_embedding'],
    'decoder_embed_layer_norm': bart['decoder']['layernorm_embedding'],
    'encoder_layers': [convert_transformer_encoder(bart['encoder']['layers'][str(i)]) for i in range(6)],
    'decoder_layers': [convert_transformer_decoder(bart['decoder']['layers'][str(i)]) for i in range(6)],
}

lm_head = params['embedding']['embedding'].T

# model architecture

def fwd_layer_norm(params: dict, x: np.ndarray, eps: float=1e-5) -> np.ndarray:
    # params
    scale: np.ndarray = params['scale']  # array
    bias: np.ndarray = params['bias']  # array

    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * scale + bias

def fwd_embedding(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    embedding: np.ndarray = params['embedding']  # array

    y = embedding[x]
    return y

def fwd_linear(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    kernel: np.ndarray = params['kernel']  # array
    bias: np.ndarray = params['bias']  # array

    return np.dot(x, kernel) + bias

def fwd_attention(params: dict, src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # params
    q_proj: dict = params['q_proj']  # linear
    k_proj: dict = params['k_proj']  # linear
    v_proj: dict = params['v_proj']  # linear
    ff: dict = params['ff']  # linear

    _, _, d_k = q_proj['kernel'].shape

    q = fwd_linear(q_proj, dst)
    k = fwd_linear(k_proj, src)
    v = fwd_linear(v_proj, src)

    qk = np.einsum('bkhm,bvhm->bhkv', q, k)
    qk = qk / np.sqrt(d_k)
    qk = np.where(mask, qk, np.NINF)
    qk = nn.softmax(qk)
    qk = np.where(mask, qk, 0)

    t = np.einsum('bhkv,bvhm->bkhm', qk, v)
    d0, d1, d2, d3 = t.shape
    t = t.reshape(d0, d1, d2 * d3)

    t = fwd_linear(ff, t)
    return t

def fwd_transformer_encoder(params: dict, src: np.ndarray, mask_enc: np.ndarray) -> np.ndarray:
    # params
    self_attn: dict = params['self_attn']  # attention
    self_attn_layer_norm: dict = params['self_attn_layer_norm']  # layer norm
    ff0: dict = params['ff0']  # linear
    ff1: dict = params['ff1']  # linear
    final_layer_norm: dict = params['final_layer_norm']  # layer norm

    src_ = src
    t = fwd_attention(self_attn, src, src, mask_enc)
    t = t + src_
    t = fwd_layer_norm(self_attn_layer_norm, t)

    t_ = t
    t = fwd_linear(ff0, t)
    t = nn.gelu(t)
    t = fwd_linear(ff1, t)
    t = t + t_
    t = fwd_layer_norm(final_layer_norm, t)
    return t

def fwd_transformer_decoder(params: dict, src: np.ndarray, dst: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray) -> np.ndarray:
    # params
    self_attn: dict = params['self_attn']  # attention
    self_attn_layer_norm: dict = params['self_attn_layer_norm']  # layer norm
    cross_attn: dict = params['cross_attn']  # attention
    cross_attn_layer_norm: dict = params['cross_attn_layer_norm']  # layer norm
    ff0: dict = params['ff0']  # linear
    ff1: dict = params['ff1']  # linear
    final_layer_norm: dict = params['final_layer_norm']  # layer norm

    dst_ = dst
    dst = fwd_attention(self_attn, dst, dst, mask_dec)
    dst = dst + dst_
    dst = fwd_layer_norm(self_attn_layer_norm, dst)

    dst_ = dst
    src = fwd_attention(cross_attn, src, dst, mask_dec_enc)
    t = src + dst_
    t = fwd_layer_norm(cross_attn_layer_norm, t)

    t_ = t
    t = fwd_linear(ff0, t)
    t = nn.gelu(t)
    t = fwd_linear(ff1, t)
    t = t + t_
    t = fwd_layer_norm(final_layer_norm, t)
    return t

def fwd_transformer(params: dict, src: np.ndarray, dst: np.ndarray, mask_enc: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray) -> np.ndarray:
    # params
    embedding: dict = params['embedding']  # embedding
    encoder_embed_positions: np.ndarray = params['encoder_embed_positions']  # array
    decoder_embed_positions: np.ndarray = params['decoder_embed_positions']  # array
    encoder_embed_layer_norm: dict = params['encoder_embed_layer_norm']  # layer norm
    decoder_embed_layer_norm: dict = params['decoder_embed_layer_norm']  # layer norm
    encoder_layers: list = params['encoder_layers']  # list of transformer encoder
    decoder_layers: list = params['decoder_layers']  # list of transformer encoder

    _, width_enc = src.shape
    _, width_dec = dst.shape

    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bart/modeling_flax_bart.py#L718-L719
    offset = 2

    src = fwd_embedding(embedding, src)
    src = src + encoder_embed_positions[offset:width_enc+offset]
    src = fwd_layer_norm(encoder_embed_layer_norm, src)
    for encoder_layer in encoder_layers:
        src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    dst = fwd_embedding(embedding, dst)
    dst = dst + decoder_embed_positions[offset:width_dec+offset]
    dst = fwd_layer_norm(decoder_embed_layer_norm, dst)
    for decoder_layer in decoder_layers:
        dst = fwd_transformer_decoder(decoder_layer, src, dst, mask_dec, mask_dec_enc)

    return dst

# inference

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

sentences = ['Can you see the beautiful flowers <mask> alongside the track?']
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

print(tokenizer.batch_decode(dst, skip_special_tokens=True))
