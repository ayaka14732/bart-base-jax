from .is_unused import is_unused

token_to_token_id = {}
token_id_to_token = {}

with open('vocab-bart-base-chinese.txt', encoding='utf-8') as f:
    for token_id, line in enumerate(f):
        token = line.rstrip('\n')

        if is_unused(token):
            continue

        token_to_token_id[token] = token_id
        token_id_to_token[token_id] = token
