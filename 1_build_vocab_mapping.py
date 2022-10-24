from collections import defaultdict
import regex as re
from transformers import BertTokenizer

from lib.utils.download_file import download_file

###########

download_file('https://raw.githubusercontent.com/StarCC0/dict/main/STCharacters.txt', 'STCharacters.txt')
download_file('https://raw.githubusercontent.com/StarCC0/dict/main/TSCharacters.txt', 'TSCharacters.txt')

###########

tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
tokenizer.save_vocabulary('vocab-bart-base-chinese.txt')

###########

pattern_should_remove = re.compile(r'##[\p{Unified_Ideograph}\u3006\u3007]')

def should_remove(token: str) -> bool:
    return bool(pattern_should_remove.fullmatch(token))

###########

token_to_token_id = {}
token_id_to_token = {}

with open('vocab-bart-base-chinese.txt', encoding='utf-8') as f:
    for token_id, line in enumerate(f):
        token = line.rstrip('\n')

        if should_remove(token):
            continue

        token_to_token_id[token] = token_id
        token_id_to_token[token_id] = token

###########

conv_table = {}

with open('STCharacters.txt', encoding='utf-8') as f:
    for line in f:
        simp, trads = line.rstrip('\n').split('\t')
        trads = ''.join(trads.split(' '))
        conv_table[simp] = trads

###########

# alpha character: characters with multiple simplified forms and one of them is the same as the original one

alpha_chars = set()

with open('TSCharacters.txt', encoding='utf-8') as f:
    for line in f:
        trad, simps = line.rstrip('\n').split('\t')
        simps = simps.split(' ')
        if trad in simps:
            alpha_chars.add(trad)

def is_alpha_char(c: str) -> bool:
    return c in alpha_chars

###########

pattern_is_cjkv = re.compile(r'[\p{Unified_Ideograph}\u3006\u3007]')

def is_cjkv(token: str) -> bool:
    return bool(pattern_is_cjkv.fullmatch(token))

###########

token_new_to_candidate_token_ids = defaultdict(set)

for token_id, token in token_id_to_token.items():
    if not is_cjkv(token):  # non-CJKV
        token_new_to_candidate_token_ids[token].add(token_id)

    else:  # CJKV
        trads = conv_table.get(token)

        if trads is None:
            token_new_to_candidate_token_ids[token].add(token_id)
        elif len(trads) == 1:
            trad = trads
            token_new_to_candidate_token_ids[trad].add(token_id)
        else:
            trad_first, *trad_rests = trads

            # trad_first
            token_new_to_candidate_token_ids[trad_first].add(token_id)

            # trad_rests
            for trad_rest in trad_rests:
                if trad_rest in token_to_token_id:
                    token_id_new = token_to_token_id[trad_rest]
                else:
                    token_id_new = token_id
                token_new_to_candidate_token_ids[trad_rest].add(token_id_new)

###########

def filter_candidate_token_ids(token_new: str, candidate_token_ids: set[int]) -> set[int]:
    # non-CJKV tokens

    if not is_cjkv(token_new):
        return candidate_token_ids

    # CJKV tokens with length of 1

    if len(candidate_token_ids) == 1:
        return candidate_token_ids

    # CJKV tokens with length greater than 1

    candidate_token_ids_new = set()

    for candidate_token_id in candidate_token_ids:
        candidate_token = token_id_to_token[candidate_token_id]

        if not is_alpha_char(token_new) and candidate_token == token_new:
            continue

        candidate_token_ids_new.add(candidate_token_id)

    return candidate_token_ids_new

token_new_to_candidate_token_ids = {
    token_new: filter_candidate_token_ids(token_new, candidate_token_ids)
    for token_new, candidate_token_ids
    in token_new_to_candidate_token_ids.items()
}

###########

for token_new, candidate_token_ids in token_new_to_candidate_token_ids.items():
    if len(candidate_token_ids) == 1:
        token_id_new = next(iter(candidate_token_ids))

    elif len(candidate_token_ids) > 1:
        candidate_tokens = [
            token_id_to_token[candidate_token_id]
            for candidate_token_id
            in candidate_token_ids
        ]

        # print(token_new, ''.join(candidate_tokens))  # for debug

        preferences = {
            '麼': '么',  # 么麽
            '於': '于',  # 於于
            '夥': '伙',  # 伙夥
            '餘': '余',  # 余馀
            '徵': '征',  # 徵征
            '鍾': '钟',  # 钟锺
            '諮': '咨',  # 咨谘
            '麪': '面',  # 麺面
        }
        candidate_token = preferences[token_new]  # guaranteed that `token_new` is always inside `preferences`
        token_id_new = token_to_token_id[candidate_token]

    else:  # len(candidate_token_ids) == 0
        raise ValueError('The length of `candidate_token_ids` should not be zero.')

    print(token_new, token_id_new)
