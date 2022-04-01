from flax.serialization import msgpack_serialize, msgpack_restore
import jax
import json
import jax.numpy as np
import numpy as onp
from transformers import BartForConditionalGeneration

# Option 1: `msgpack_serialize` and `msgpack_restore`
# Option 2: `pickle.dumps` and `pickle.loads`
serialize = msgpack_serialize
deserialize = msgpack_restore

model = BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')
bart = dict(model.model.named_parameters())

for k in bart:
    if '.' in k:
        ks = k.split('.')
        dic_pointer = bart
        for nk in ks:
            if nk not in dic_pointer:
                dic_pointer[nk]={}
            dic_pointer = dic_pointer[nk]
        final_k = ks[-1]
        dic_pointer[final_k] = bart[k]
        bart.pop(k)


def convert_qkv(params):
    return {
        'kernel': params['weight'].reshape(768, 12, 64).transpose(1, 0, 2),
        'bias': params['bias'].reshape(12, 64),
    }

def convert_linear(params):
    return{
        'kernel': params['weight'],
        'bias': params['bias']
    }

def convert_layer_norm(params):
    return {
        'scale': params['weight'],
        'bias': params['bias']
    }

def convert_transformer_encoder(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv(params['self_attn']['q_proj']),
            'k_proj': convert_qkv(params['self_attn']['k_proj']),
            'v_proj': convert_qkv(params['self_attn']['v_proj']),
            'ff': convert_linear(params['self_attn']['out_proj']),
        },
        'self_attn_layer_norm': convert_layer_norm(params['self_attn_layer_norm']),
        'ff0': convert_linear(params['fc1']),
        'ff1': convert_linear(params['fc2']),
        'final_layer_norm': convert_layer_norm(params['final_layer_norm']),
    }

# def convert_transformer_decoder(params):
#     return {
#         'self_attn': {
#             'q_proj': convert_qkv(params['self_attn']['q_proj']),
#             'k_proj': convert_qkv(params['self_attn']['k_proj']),
#             'v_proj': convert_qkv(params['self_attn']['v_proj']),
#             'ff': params['self_attn']['out_proj'],
#         },
#         'self_attn_layer_norm': params['self_attn_layer_norm'],
#         'cross_attn': {
#             'q_proj': convert_qkv(params['encoder_attn']['q_proj']),
#             'k_proj': convert_qkv(params['encoder_attn']['k_proj']),
#             'v_proj': convert_qkv(params['encoder_attn']['v_proj']),
#             'ff': params['encoder_attn']['out_proj'],
#         },
#         'cross_attn_layer_norm': params['encoder_attn_layer_norm'],
#         'ff0': params['fc1'],
#         'ff1': params['fc2'],
#         'final_layer_norm': params['final_layer_norm'],
#     }

params = {
    'embedding': {'embedding': bart['shared.weight']},
    'encoder_embed_positions': bart['encoder']['embed_positions']['embedding'],
    # 'decoder_embed_positions': bart['decoder']['embed_positions']['embedding'],
    'encoder_embed_layer_norm': bart['encoder']['layernorm_embedding'],
    # 'decoder_embed_layer_norm': bart['decoder']['layernorm_embedding'],
    'encoder_layers': [convert_transformer_encoder(bart['encoder']['layers'][str(i)]) for i in range(6)],
    # 'decoder_layers': [convert_transformer_decoder(bart['decoder']['layers'][str(i)]) for i in range(6)],
}

arr2shape = lambda x: str(x.shape).replace(',)', ')')
d = jax.tree_map(arr2shape, params)
print(json.dumps(d, indent=2))

serialized_params = serialize(params)
recovered_params = deserialize(serialized_params)

def assert_tree_equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(a, onp.ndarray):
        assert isinstance(b, np.ndarray) or isinstance(b, onp.ndarray), f'{type(b)}'
        assert np.allclose(a, b)

    elif isinstance(a, dict):
        assert isinstance(b, dict), f'{type(b)}'
        keys_a = sorted(a)
        keys_b = sorted(b)
        assert keys_a == keys_b
        for key in keys_a:
            assert_tree_equal(a[key], b[key])

    elif isinstance(a, list):
        assert isinstance(b, list), f'{type(b)}'
        assert len(a) == len(b)
        for a_, b_ in zip(a, b):
            assert_tree_equal(a_, b_)

    else:
        raise NotImplementedError(f'Unsupported element type: {type(a)}')

assert_tree_equal(params, recovered_params)

with open('bart_ch_params.dat', 'wb') as f:
    f.write(serialized_params)
