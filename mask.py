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
         佛祖保佑         永無 BUG
'''

proposed_mask_rate = 0.188  # actual mask rate would be approximately 0.15
poisson_rate = 4.2  # span length = 3 would be the most frequent in the actual distribution
max_span_len = 10
random.seed(seed)

MaskScheme = list[tuple[int, int]]

def normalise_probs(a: np.ndarray) -> np.ndarray:
    return a / a.sum()

def generate_probs_list() -> list[list[float]]:
    probs_list = []

    poisson = dist.Poisson(rate=poisson_rate)
    probs = np.exp(poisson.log_prob(np.arange(max_span_len + 1)))

    probs_ = normalise_probs(probs)
    probs_list.append(probs_.cumsum().tolist())

    for i in range(max_span_len - 1):
        probs_ = normalise_probs(probs[:-i-1])
        probs_list.append(probs_.cumsum().tolist())

    return probs_list[::-1]

probs_list = generate_probs_list()

def determine_should_mask_len(seq_len: int) -> int:
    x = seq_len * proposed_mask_rate
    integer_part = int(x)
    fractional_part = x - float(integer_part)
    should_add = random.random() < fractional_part
    should_mask_len = integer_part + should_add
    return should_mask_len

def generate_spans(should_mask_len: int) -> list[int]:
    spans = []
    while should_mask_len > 0:
        current_max_span_len = min(max_span_len, should_mask_len)
        probs = probs_list[current_max_span_len - 1]
        span_len = random.choices(range(current_max_span_len + 1), cum_weights=probs)[0]
        spans.append(span_len)
        should_mask_len -= span_len + 1
    random.shuffle(spans)
    return spans

def distribute_insert_poses(abs_insert_poses: list[int], spans: list[int]) -> MaskScheme:
    offset = 0
    mask_scheme = []
    for abs_insert_pos, span in zip(abs_insert_poses, spans):
        insert_pos = abs_insert_pos + offset
        mask_scheme.append((insert_pos, span))
        offset += span + 1
    return mask_scheme

def random_add_one(mask_scheme: MaskScheme) -> MaskScheme:
    should_add_one = random.random() < 0.5
    if should_add_one:
        mask_scheme = [(insert_pos + 1, span) for insert_pos, span in mask_scheme]
    return mask_scheme

def generate_mask_scheme(seq_len: int) -> MaskScheme:
    should_mask_len = determine_should_mask_len(seq_len)
    spans = generate_spans(should_mask_len)

    n_spans = len(spans)
    n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
    abs_insert_poses = sorted(random.sample(range(n_possible_insert_poses), n_spans))

    mask_scheme = distribute_insert_poses(abs_insert_poses, spans)
    mask_scheme = random_add_one(mask_scheme)
    return mask_scheme

def test():
    def pretty_print_mask_scheme(seq_len: int, mask_scheme: MaskScheme) -> None:
        x = ['.'] * seq_len

        for insert_pos, span in mask_scheme:
            for i in range(insert_pos, insert_pos + span):
                x[i] = 'x'

        offset = 0
        for insert_pos, span in mask_scheme:
            x.insert(insert_pos + offset, '(')
            offset += 1
            x.insert(insert_pos + span + offset, ')')
            offset += 1

        s = ''.join(x)
        assert ')(' not in s, 'two masks cannot be continuous'
        print(s)

    def pretty_print_array(a: np.ndarray) -> str:
        return f'[{", ".join(map(lambda x: f"{x:.4f}", a))}]'

    def undo_cumsum(a: np.ndarray) -> np.ndarray:
        return np.diff(a, prepend=0)

    total_seq_len = 0
    total_mask_len = 0
    span_lens = [0] * (max_span_len + 1)

    for _ in range(200000):
        seq_len = random.randrange(2, 256)
        mask_scheme = generate_mask_scheme(seq_len)

        # print(mask_scheme)
        # pretty_print_mask_scheme(seq_len, mask_scheme)

        mask_len = sum(span for _, span in mask_scheme)
        total_seq_len += seq_len
        total_mask_len += mask_len
        for _, span in mask_scheme:
            span_lens[span] += 1

    print(f'Proposed mask rate: {proposed_mask_rate:.2%}')
    print(f'Actual mask rate: {total_mask_len / total_seq_len:.2%}')
    print(f'Proposed span length distribution: {pretty_print_array(undo_cumsum(np.array(probs_list[-1])))}')
    print(f'Actual span length distribution: {pretty_print_array(normalise_probs(np.array(span_lens)))}')

if __name__ == '__main__':
    test()
