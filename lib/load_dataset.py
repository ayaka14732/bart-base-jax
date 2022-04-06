import math
from multiprocessing import Pool
import numpy as np
from os.path import expanduser, join
from tqdm import tqdm
from transformers import BartTokenizer, BertTokenizer

def chunks(lst, n):
    '''Yield successive n-sized chunks from lst.'''
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
ch_tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')

max_len_en = 512 - 1  # prepend EOS at the beginning
max_len_zh = 256
chunksize = 1000

def encode_one_batch(sents):
    zh_sents = []
    en_sents = []

    for zh_sent, en_sent in sents:
        zh_sents.append(zh_sent)
        en_sents.append(en_sent)

    output_zh = ch_tokenizer(zh_sents, return_tensors='np', max_length=max_len_zh, padding='max_length', truncation=True)
    output_en = en_tokenizer(en_sents, return_tensors='np', max_length=max_len_en, padding='max_length', truncation=True)

    data_zh = output_zh.input_ids
    mask_zh = output_zh.attention_mask

    data_en = output_en.input_ids
    mask_en = output_en.attention_mask

    data_en = np.hstack(([[2]] * len(sents), data_en))
    mask_en = np.hstack(([[1]] * len(sents), mask_en))

    mask_zh = mask_zh.astype(np.bool_)
    mask_en = mask_en.astype(np.bool_)

    return data_zh, mask_zh, data_en, mask_en

def load_dataset(dataset_ch, dataset_en, max_length=None):
    with open(join(expanduser(f'~/dataset/processed'), dataset_ch), encoding='utf-8') as fzh, open(join(expanduser(f'~/dataset/processed'), dataset_en), encoding='utf-8') as fen:
        filtered_sentences = []
        count = 0

        for line_zh, line_en in zip(fzh, fen):
            line_zh = line_zh.rstrip('\n')
            line_en = line_en.rstrip('\n')

            # skip long sentences
            if len(line_zh) <= max_len_zh - 2 and len(line_en.split()) <= max_len_en - 2:
                filtered_sentences.append((line_zh, line_en))

                count += 1
                if max_length is not None and count == max_length:
                    break

    with Pool() as p:
        xs = list(tqdm(p.imap(encode_one_batch, chunks(filtered_sentences, chunksize)), total=math.ceil(len(filtered_sentences) / chunksize)))

    data_list_zh = []
    mask_list_zh = []
    data_list_en = []
    mask_list_en = []

    for data_zh, mask_zh, data_en, mask_en in xs:
        data_list_zh.append(data_zh)
        mask_list_zh.append(mask_zh)
        data_list_en.append(data_en)
        mask_list_en.append(mask_en)

    data_list_zh = np.vstack(data_list_zh)
    mask_list_zh = np.vstack(mask_list_zh)
    data_list_en = np.vstack(data_list_en)
    mask_list_en = np.vstack(mask_list_en)

    return data_list_zh, mask_list_zh, data_list_en, mask_list_en
