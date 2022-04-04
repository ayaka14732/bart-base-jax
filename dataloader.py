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

en_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
ch_tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')

max_length_en = 512
max_length_ch = 256
chunksize = 1000

def encode_en_batch(sents):
    #assert isinstance(sents, list)
    #assert len(sents) > 1
    y = en_tokenizer(sents, return_tensors='np', max_length=max_length_en, padding='max_length', truncation=True)
    data = y.input_ids
    mask = y.attention_mask
    return data, mask

def encode_ch_batch(sents):
    #assert isinstance(sents, list)
    #assert len(sents) > 1
    y = ch_tokenizer(sents, return_tensors='np', max_length=max_length_ch, padding='max_length',truncation=True)
    data = y.input_ids
    mask = y.attention_mask
    return data, mask


def process_one_dataset(dataset_ch, dataset_en):
    with open(join(expanduser(f'~/dataset/processed'), dataset_ch), encoding='utf-8') as fch:
        with open(join(expanduser(f'~/dataset/processed'), dataset_en), encoding='utf-8') as fen:
            zh = []
            en = []
            # count = 0
            for  linech, lineen in zip(fch, fen):
                linech = linech.rstrip('\n')
                lineen = lineen.rstrip('\n')
                if len(linech) <= max_length_ch - 2 and len(lineen.split()) <= max_length_en - 2:  # skip long sentences
                    zh.append(linech)
                    en.append(lineen)
                    # count+=1
                    # if count==960:
                    #     break

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



#ch_tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")

# input_ids, mask_enc_1d, decoder_input_ids, mask_dec_1d = process_one_dataset('wikimatrix21.zh', 'wikimatrix21.en')
#np.savez("wikimatrix21zh.npz", ch_data = data_arr_ch, ch_mask = data_arr_ch)
# print("Done1")
# del data_arr_en, mask_arr_en, data_arr_ch, mask_arr_ch
#
# data_arr_en, mask_arr_en, data_arr_ch, mask_arr_ch = process_one_dataset('UN21.zh', 'UN21.en')
# print("Done2")
# #np.savez("UN21en.npz", ch_data = data_arr, ch_mask = mask_arr)
# #print("Done22")
# del data_arr_en, mask_arr_en, data_arr_ch, mask_arr_ch
#
# data_arr_en, mask_arr_en, data_arr_ch, mask_arr_ch = process_one_dataset('Wikititles21.zh', 'Wikititles21.en')
# print("Done3")
# #np.savez("wikititles21zh.npz", ch_data = data_arr, ch_mask = mask_arr)
# del data_arr_en, mask_arr_en, data_arr_ch, mask_arr_ch
#
# max_length_en = 256
# max_length_ch = 128
# ddata_arr_en, mask_arr_en, data_arr_ch, mask_arr_ch = process_one_dataset('backtr.zh', 'backtr.en')
# #np.savez("backtranslatednewszh.npz", ch_data = data_arr, ch_mask = mask_arr)
# print("Done4")
# del data_arr_en, mask_arr_en, data_arr_ch, mask_arr_ch



#/home/sarun/dataset/processed/backtranslatednews.en


'''
from transformers import BertTokenizer
import numpy as np
from tempfile import TemporaryFile

import csv

en_tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base")

with open("/home/sarun/dataset/processed/wikimatrix21en.txt", encoding='utf-8') as file:
    wikimatrix21en = file.read().splitlines()

batch = en_tokenizer(wikimatrix21en, return_tensors = 'np', max_length = 1024, padding = 'max_length')
en_data = batch["input_ids"]
en_mask = batch["attention_mask"]
np.savez("wikimatrix21en.npz", en_data = en_data, en_mask = en_mask)

with open("/home/sarun/dataset/processed/UN21en.txt", encoding='utf-8') as file:
    un211en = file.read().splitlines()

batch = en_tokenizer(wikimatrix21en, return_tensors = 'np', max_length = 1024, padding = 'max_length')
en_data = batch["input_ids"]
en_mask = batch["attention_mask"]
np.savez("UN21en.npz", en_data = en_data, en_mask = en_mask)

with open("/home/sarun/dataset/processed/wikititles21en.txt", encoding='utf-8') as file:
    wikititles21en = file.read().splitlines()

batch = en_tokenizer(wikimatrix21en, return_tensors = 'np', max_length = 128, padding = 'max_length')
en_data = batch["input_ids"]
en_mask = batch["attention_mask"]
np.savez("wikititles21en.npz", en_data = en_data, en_mask = en_mask)

with open("/home/sarun/dataset/processed/backtranslatednews.en", encoding='utf-8') as file:
    backtren = file.read().splitlines()

batch = en_tokenizer(backtren, return_tensors = 'np', max_length = 1024, padding = 'max_length')
en_data = batch["input_ids"]
en_mask = batch["attention_mask"]
np.savez("backtranslatednews.npz", en_data = en_data, en_mask = en_mask)'''