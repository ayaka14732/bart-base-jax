import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import jax.numpy as np
import regex as re
import sys
from transformers import MarianTokenizer, FlaxMarianMTModel
from tqdm import tqdm
from typing import Any

from lib.dataset.load_cantonese import load_cantonese
from lib.param_utils.load_params import load_params

def chunks(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def post_process(s: str) -> str:
    '''
    >>> remove_space('阿 爸 好 忙 ， 成 日 出 差')
    '阿爸好忙，成日出差'
    >>> remove_space('摸 A B 12至 3')
    '摸A B 12至3'
    >>> remove_space('噉你哋要唔要呢 ？')
    '噉你哋要唔要呢？'
    >>> remove_space('3 . 1')
    '3.1'
    '''
    s = re.sub(r'(?<=[\p{Unified_Ideograph}\u3006\u3007。，、！：？（）《》「」]) (?=[\p{Unified_Ideograph}\u3006\u3007。，、！：？（）《》「」])', r'', s)
    s = re.sub(r'(?<=[\p{Unified_Ideograph}\u3006\u3007。，、！：？（）《》「」]) (?=[\da-zA-Z])', r'', s)
    s = re.sub(r'(?<=[\da-zA-Z]) (?=[\p{Unified_Ideograph}\u3006\u3007。，、！：？（）《》「」])', r'', s)
    s = re.sub(r'(?<=[\da-zA-Z]) (?=[.,])', r'', s)
    s = re.sub(r'(?<=[.,]) (?=[\da-zA-Z])', r'', s)
    s = s.replace(',', '，')
    s = s.replace('?', '？')
    s = s.replace('!', '！')
    return s

sentences = load_cantonese(split='test')
sentences_en = [en for en, _ in sentences]

param_file = sys.argv[1] if len(sys.argv) >= 2 else 'absurd-breeze-8-5.dat'
params = load_params(param_file)
params = jax.tree_map(np.asarray, params)

model = FlaxMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
tokenizer_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh', source_spm='source.spm')
tokenizer_yue = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh', source_spm='target.spm')

predictions = []

for chunk in tqdm(chunks(sentences_en, chunk_size=32)):
    inputs = tokenizer_en(chunk, return_tensors='jax', padding=True)
    generated_ids = model.generate(**inputs, num_beams=5, max_length=128, params=params).sequences
    decoded_sentences = tokenizer_yue.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for decoded_sentence in decoded_sentences:
        predictions.append(decoded_sentence)

with open('results-baseline.txt', 'w', encoding='utf-8') as f:
    for prediction in predictions:
        prediction = post_process(prediction)
        print(prediction, file=f)
