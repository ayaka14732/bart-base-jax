import flax.linen as nn
import jax.numpy as np

def random_init_embed(key, num: int):
    embed = nn.Embed(num_embeddings=num, features=768)
    embed_params = embed.init(key, np.ones((2,), dtype=np.uint32))['params']['embedding']
    return embed_params
