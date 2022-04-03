import math
from multiprocessing import Pool
from xmlrpc.client import boolean
import numpy as np
from os.path import expanduser, join
from tqdm import tqdm
from transformers import BertTokenizer, BartTokenizer

def chunks(lst, n):
    '''Yield successive n-sized chunks from lst.'''
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

chunksize = 1000

ch_tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')

en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def encode_one_batch_ch(sents):
    assert isinstance(sents, list)
    assert len(sents) > 1
    y = ch_tokenizer(sents, return_tensors='np', max_length=256, padding='max_length')
    data = y.input_ids
    mask = y.attention_mask
    return data, mask

def encode_one_batch_en(sents):
    assert isinstance(sents, list)
    assert len(sents) > 1
    y = en_tokenizer(sents, return_tensors='np', max_length=512, padding='max_length')
    data = y.input_ids
    mask = y.attention_mask
    return data, mask


def process_one_dataset(dataset_filename, language):
    if language == 'en':
        encode_one_batch = encode_one_batch_ch
        max_length = 256
    elif language == 'zh':
        encode_one_batch = encode_one_batch_en
        max_length = 512
    else:
        print('unsupported language specification ' + language)
        exit(0)


    with open(join(expanduser(f'~/dataset/processed'), dataset_filename), encoding='utf-8') as f:
        raw = []
        for i,line in enumerate(f):
            line = line.rstrip('\n')
            # if len(line) <= max_length - 2:  # skip long sentences
            raw.append(line)

    with Pool() as p:
        xs = list(tqdm(p.imap(encode_one_batch,chunks(raw, chunksize)), total=math.ceil(len(raw) / chunksize)))

    data_list = []
    mask_list = []

    for data, mask in xs:
        data_list.append(data)
        mask_list.append(mask)

    data_arr = np.vstack(data_list)
    mask_arr = np.vstack(mask_list)

    return data_arr, mask_arr.astype(boolean)


# data_arr, mask_arr = process_one_dataset('wikimatrix21zh.txt')
# #np.savez("wikimatrix21zh.npz", ch_data = data_arr, ch_mask = mask_arr)
# print("wikimatrix completed")

#
# data_arr, mask_arr = process_one_dataset('UN21.zh')
# #np.savez("UN21zh.npz", ch_data = data_arr, ch_mask = mask_arr)
# print("UN completed")
#
#
#
# data_arr, mask_arr = process_one_dataset('backtranslatednews.zh')
# #np.savez("backtranslatednewszh.npz", ch_data = data_arr, ch_mask = mask_arr)
# print("backtranslated completed")
# #/home/sarun/dataset/processed/backtranslatednews.en
#
# max_length = 128
#
# data_arr, mask_arr = process_one_dataset('wikititles21zh.txt')
# #np.savez("wikititles21zh.npz", ch_data = data_arr, ch_mask = mask_arr)
# print("wikititles completed")

