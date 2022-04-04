from transformers import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')
model_trained = dict(model.model.named_parameters())

import jax.numpy as np
import numpy as onp

def dotted_dict2nested_dict(params):
    for k in list(params):
        if 'decoder' in k:
            params.pop(k)
            continue
        if '.' in k:
            ks = k.split('.')
            dic_pointer = params
            for nk in ks[:-1]:
                if nk not in dic_pointer:
                    dic_pointer[nk]={}
                dic_pointer = dic_pointer[nk]
            final_k = ks[-1]
            dic_pointer[final_k] = np.asarray(params[k].detach().numpy())
            params.pop(k)
    return params

bart_trained = dotted_dict2nested_dict(model_trained)

def convert_qkv(params):
    return {
        'kernel': params['kernel'].reshape(768, 12, 64).transpose(1, 0, 2),
        'bias': params['bias'].reshape(12, 64),
    }

def convert_qkv_pt(params):
    return {
        'kernel': params['weight'].T.reshape(768, 12, 64).transpose(1, 0, 2),
        'bias': params['bias'].reshape(12, 64),
    }

def convert_linear_pt(params):
    return {
        'kernel': params['weight'].T,
        'bias': params['bias'],
    }

def convert_layer_norm_pt(params):
    return {
        'scale': params['weight'],
        'bias': params['bias'],
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

def convert_transformer_encoder_pt(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv_pt(params['self_attn']['q_proj']),
            'k_proj': convert_qkv_pt(params['self_attn']['k_proj']),
            'v_proj': convert_qkv_pt(params['self_attn']['v_proj']),
            'ff': convert_linear_pt(params['self_attn']['out_proj']),
        },
        'self_attn_layer_norm': convert_layer_norm_pt(params['self_attn_layer_norm']),
        'ff0': convert_linear_pt(params['fc1']),
        'ff1': convert_linear_pt(params['fc2']),
        'final_layer_norm': convert_layer_norm_pt(params['final_layer_norm']),
    }

params_trained = {
    'embedding': {'embedding': bart_trained['shared']['weight']},
    'encoder_embed_positions': bart_trained['encoder']['embed_positions']['weight'],
    'encoder_embed_layer_norm': convert_layer_norm_pt(bart_trained['encoder']['layernorm_embedding']),
    'encoder_layers': [convert_transformer_encoder_pt(bart_trained['encoder']['layers'][str(i)]) for i in range(6)],
}

from transformers import BartConfig, FlaxBartModel

config = BartConfig.from_pretrained('fnlp/bart-base-chinese')

bart_untrained = FlaxBartModel(config=config, seed=42).params

params_untrained = {
    'embedding': {'embedding': bart_untrained['shared']['embedding']},
    'encoder_embed_positions': bart_untrained['encoder']['embed_positions']['embedding'],
    'encoder_embed_layer_norm': bart_untrained['encoder']['layernorm_embedding'],
    'encoder_layers': [convert_transformer_encoder(bart_untrained['encoder']['layers'][str(i)]) for i in range(6)],
}

import jax

# check
assert str(jax.tree_map(lambda x: x.shape, params_trained)) == \
       str(jax.tree_map(lambda x: x.shape, params_untrained))

# Model 1: randomly initialize all params

model_1 = params_untrained

# Model 2: initialize pre-trained Chinese embedding and encoder_embed_layer_norm,
# randomly initialize others

model_2 = {
    'embedding': params_trained['embedding'],
    'encoder_embed_positions': params_trained['encoder_embed_positions'],
    'encoder_embed_layer_norm': params_trained['encoder_embed_layer_norm'],
    'encoder_layers': params_untrained['encoder_layers'],
}

# Model 3: initialize pre-trained Chinese embedding and encoder_embed_layer_norm,
# initialize the first three layers of transformer,
# randomly initialize others

model_3 = {
    'embedding': params_trained['embedding'],
    'encoder_embed_positions': params_trained['encoder_embed_positions'],
    'encoder_embed_layer_norm': params_trained['encoder_embed_layer_norm'],
    'encoder_layers': [
        params_trained['encoder_layers'][0],
        params_trained['encoder_layers'][1],
        params_trained['encoder_layers'][2],
        params_untrained['encoder_layers'][3],
        params_untrained['encoder_layers'][4],
        params_untrained['encoder_layers'][5],
    ],
}

from flax.serialization import msgpack_serialize, msgpack_restore

# Option 1: `msgpack_serialize` and `msgpack_restore`
# Option 2: `pickle.dumps` and `pickle.loads`
serialize = msgpack_serialize
deserialize = msgpack_restore

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

def write_params_to_file(params, filename):
    serialized_params = serialize(params)
    recovered_params = deserialize(serialized_params)

    assert_tree_equal(params, recovered_params)

    with open(filename, 'wb') as f:
        f.write(serialized_params)

write_params_to_file(model_1, 'ch_untrained.dat')
write_params_to_file(model_2, 'ch_trained_emb.dat')
write_params_to_file(model_3, 'ch_trained_emb_3_layers.dat')
