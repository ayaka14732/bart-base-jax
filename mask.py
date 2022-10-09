import jax; jax.config.update('jax_platforms', 'cpu')
import jax.numpy as np
import numpyro.distributions as dist
import random

seed = r'''
                  _oo0oo_
                 o8888888o
                 88" . "88
                 (| -_- |)
                 0\  =  /0
               ___/`---'\___
             .' \\|     | '.
            / \\|||  :  ||| \
           / _||||| -:- |||||- \
          |   | \\\  -  / |   |
          | \_|  ''\---/''  |_/ |
          \  .-\__  '-'  ___/-. /
        ___'. .'  /--.--\  `. .'___
     ."" '<  `.___\_<|>_/___.' >' "".
    | | :  `- \`.;`\ _ /`;.`/ - ` : | |
    \  \ `_.   \_ __\ /__ _/   .-` /  /
=====`-.____`.___ \_____/___.-`___.-'=====
                  `=---='
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         佛祖保佑         永无 BUG
'''

mask_rate = 0.15
poisson_rate = 3.5
max_span_len = 10
random.seed(seed)

def normalise_probs(a):
    return a / a.sum()

def generate_probs_list():
    probs_list = []

    poisson = dist.Poisson(rate=poisson_rate)
    probs = np.exp(poisson.log_prob(np.arange(max_span_len + 1)))

    probs_ = normalise_probs(probs)
    probs_list.append(tuple(probs_.cumsum().tolist()))

    for i in range(max_span_len - 1):
        probs_ = normalise_probs(probs[:-i-1])
        probs_list.append(tuple(probs_.cumsum().tolist()))

    return tuple(probs_list[::-1])

probs_list = generate_probs_list()

def random_insert(xs, a):
    xs.insert(random.randrange(len(xs) + 1), a)

def determine_should_mask_len(seq_len):
    x = seq_len * mask_rate
    integer_part = int(x)
    fractional_part = x - float(integer_part)
    should_add = random.random() < fractional_part
    should_mask_len = integer_part + should_add
    return should_mask_len

def generate_spans(should_mask_len):
    spans = []
    while should_mask_len > 0:
        current_max_span_len = min(max_span_len, should_mask_len)
        probs = probs_list[current_max_span_len - 1]
        span_len = random.choices(range(current_max_span_len + 1), cum_weights=probs)[0]
        spans.append(span_len)
        should_mask_len -= span_len + 1
    random.shuffle(spans)
    return spans

def distribute_insert_poses(abs_insert_poses, spans):
    offset = 0
    mask_scheme = []
    for abs_insert_pos, span in zip(abs_insert_poses, spans):
        insert_pos = abs_insert_pos + offset
        mask_scheme.append((insert_pos, span))
        offset += span + 1
    return mask_scheme

def random_add_one(mask_scheme):
    should_add_one = random.random() < 0.5
    if should_add_one:
        mask_scheme = [(insert_pos + 1, span) for insert_pos, span in mask_scheme]
    return mask_scheme

def generate_mask_scheme(seq_len):
    should_mask_len = determine_should_mask_len(seq_len)
    spans = generate_spans(should_mask_len)

    n_spans = len(spans)
    n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
    abs_insert_poses = sorted(random.sample(range(n_possible_insert_poses), n_spans))

    mask_scheme = distribute_insert_poses(abs_insert_poses, spans)
    mask_scheme = random_add_one(mask_scheme)
    return mask_scheme

def pretty_print(seq_len, scheme):
    x = ['.'] * seq_len

    for insert_pos, span in scheme:
        for i in range(insert_pos, insert_pos + span):
            x[i] = 'x'

    offset = 0
    for insert_pos, span in scheme:
        x.insert(insert_pos + offset, '(')
        offset += 1
        x.insert(insert_pos + span + offset, ')')
        offset += 1

    s = ''.join(x)
    assert ')(' not in s, 'two masks cannot be continuous'
    print(s)

def test():
    for _ in range(20000):
        seq_len = random.randrange(2, 102)
        mask_scheme = generate_mask_scheme(seq_len)
        pretty_print(seq_len, mask_scheme)

if __name__ == '__main__':
    test()
