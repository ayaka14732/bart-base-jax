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

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
ch_tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')

max_length_en = 511
max_length_ch = 256
chunksize = 1000

def encode_en_batch(sents):
    y = en_tokenizer(sents, return_tensors='np', max_length=max_length_en, padding='max_length', truncation=True)
    data = np.hstack(([[2]] * len(sents), y.input_ids))
    mask = np.hstack(([[1]] * len(sents), y.attention_mask))
    return data, mask

def encode_ch_batch(sents):
    y = ch_tokenizer(sents, return_tensors='np', max_length=max_length_ch, padding='max_length', truncation=True)
    data = y.input_ids
    mask = y.attention_mask
    return data, mask

def process_one_dataset(dataset_ch, dataset_en, max_length=None):
    with open(join(expanduser(f'~/dataset/processed'), dataset_ch), encoding='utf-8') as fch, \
    open(join(expanduser(f'~/dataset/processed'), dataset_en), encoding='utf-8') as fen:
        zh = []
        en = []

        for i, (linech, lineen) in enumerate(zip(fch, fen)):
            linech = linech.rstrip('\n')
            lineen = lineen.rstrip('\n')
            if len(linech) <= max_length_ch - 2 and len(lineen.split()) <= max_length_en - 2:  # skip long sentences
                zh.append(linech)
                en.append(lineen)
                if max_length is not None and i == max_length:
                    break

    with Pool() as p:
        xs = list(tqdm(p.imap(encode_en_batch, chunks(en, chunksize)), total=math.ceil(len(en) / chunksize)))
        ys = list(tqdm(p.imap(encode_ch_batch, chunks(zh, chunksize)), total=math.ceil(len(zh) / chunksize)))

    data_list_en = []
    mask_list_en = []
    data_list_ch = []
    mask_list_ch = []

    for data, mask in xs:
        data_list_en.append(data)
        mask_list_en.append(mask)

    for data, mask in ys:
        data_list_ch.append(data)
        mask_list_ch.append(mask)

    data_arr_en = np.vstack(data_list_en)
    mask_arr_en = np.vstack(mask_list_en)
    data_arr_ch = np.vstack(data_list_ch)
    mask_arr_ch = np.vstack(mask_list_ch)

    return data_arr_ch, mask_arr_ch.astype(np.bool_), data_arr_en, mask_arr_en.astype(np.bool_)
