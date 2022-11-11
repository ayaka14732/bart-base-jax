import jax.numpy as np
from transformers import MarianTokenizer

tokenizer_en = tokenizer_yue = None

def tokenization_worker(sentences: list[tuple[str, str]]) -> np.ndarray:
    global tokenizer_en, tokenizer_yue
    if tokenizer_en is None:
        tokenizer_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh', source_spm='source.spm')
        tokenizer_yue = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh', source_spm='target.spm')

    sentences_en = []
    sentences_yue = []
    for sentence_en, sentence_yue in sentences:
        sentences_en.append(sentence_en)
        sentences_yue.append(sentence_yue)

    max_length = 100
    src_inputs = tokenizer_en(sentences_en, max_length=max_length, return_tensors='np', truncation=True, verbose=True, padding='max_length')
    dst_inputs = tokenizer_yue(sentences_yue, max_length=max_length-1, return_tensors='np', truncation=True, verbose=True, padding='max_length')
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
