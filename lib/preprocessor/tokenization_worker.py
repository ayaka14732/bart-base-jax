import jax.numpy as np
from transformers import MarianTokenizer

tokenizer = None

def tokenization_worker(sentences: list[tuple[str, str]]) -> np.ndarray:
    global tokenizer
    if tokenizer is None:
        tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')

    sentences_en = []
    sentences_yue = []
    for sentence_en, sentence_yue in sentences:
        sentences_en.append(sentence_en)
        sentences_yue.append(sentence_yue)

    max_length = 128
    src_inputs = tokenizer(sentences_en, max_length=max_length, return_tensors='np', truncation=True, verbose=True, add_special_tokens=False, padding='max_length')
    dst_inputs = tokenizer(sentences_yue, max_length=max_length-1, return_tensors='np', truncation=True, verbose=True, add_special_tokens=False, padding='max_length')
    # TODO: add a reminder about these default settings:
    # - `return_tensors='np'`
    # - `add_prefix_space=True`
    # return type is a tuple, not a dict
    # return `np.uint16` and `np.bool_`

    src = src_inputs.input_ids.astype(np.uint16)
    mask_enc_1d = src_inputs.attention_mask.astype(np.bool_)
    dst = dst_inputs.input_ids.astype(np.uint16)
    mask_dec_1d = dst_inputs.attention_mask.astype(np.bool_)

    return src, mask_enc_1d, dst, mask_dec_1d
