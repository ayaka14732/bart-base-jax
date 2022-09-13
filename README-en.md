# Machine Translation Model from Mandarin to Taiwanese Hokkien

This project is an implementation of the bart-base model for machine translation from Mandarin to Taiwanese Hokkien.

This project is based on the Chinese version of the pre-trained bart-base model, [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese), and fine-tuned on the compute cluster provided by [NUS SoC](https://www.comp.nus.edu.sg/).

The project is based on the `main` branch of this repository, [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax), a JAX implementation of the bart-base model, supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Task Formulation

The problem tackled in this project is machine translation from Mandarin to Taiwanese Hokkien. Specifically, the input of the model is a Mandarin sentence in traditional Chinese, and the model is supposed to output its Taiwanese Hokkien translation in traditional Chinese:

```
Input: 你的文章寫得很好，可以去投稿了。
Output: 你的文章寫了真好，會使去投稿矣。
```

## Environment Setup

The scripts are designed to be run on a single machine with one GPU.

```sh
python3.10 -m venv ./venv
. ./venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is [_Dictionary of Frequently-Used Taiwan Minnan_](https://twblg.dict.edu.tw/holodict_new/), while the data is collected from [g0v/moedict-data-twblg](https://github.com/g0v/moedict-data-twblg/blob/master/uni/%E4%BE%8B%E5%8F%A5.csv). The advantage of using this dataset is that the sentences are drawn from dictionary examples and thus contain more distinctive words in Taiwanese Hokkien, allowing the model to learn a more complete Taiwanese Hokkien vocabulary.

The data file (`lib/twblg/data.tsv`) consists of 8366 sentences in both Mandarin and Taiwanese Hokkien. The sentences are split into train, dev and test sets with a ratio of 8:1:1. However, the dev set is not utilised at present.

## Data Preprocessing

As I am using [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) as the base model, the data fed into the model must be in simplified Chinese. Therefore, I converted the dataset from traditional Chinese to simplified Chinese using [StarCC](https://github.com/StarCC0/starcc-py).

However, due to the one-to-many problem, the accuracy of the conversion from simplified Chinese to traditional Chinese is not always guaranteed. The error comes from two aspects:

Firstly, the 

For sentences in the corpus, I keep the original traditional Chinese version alongside the converted simplified Chinese version. Therefore, when the traditional Chinese version is needed, we can always retrieve it in its original form.

For new sentences (e.g. the model output), I cached a convertion table when converting from traditional Chinese to simplified Chinese. When converting back, words in the conversion table will also be utilised. For example, 代誌 ('affair') is a Taiwanese Hokkien word that is not exist in the original conversion table of StarCC, which is tailored for Mandarin. Unfortunately, the charater 志 in simplified Chinese corresponds to 志 and 誌 in traditional Chinese, and by default the StarCC module will convert it to the former, which is wrong.

## Tokeniser

The original tokeniser used in [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) is the Chinese version of the `BertTokenizer` in Hugging Face Transformer repository, with a vocabulary size of 21128. However, there are three problems with the tokeniser:

Firstly, the BERT tokeniser employs WordPiece as the subword tokenisation algorithm, while for Chinese texts, a characted-based tokenisation algorithm is already sufficient. Worse still, as the Chinese BERT tokeniser is fit on Chinese texts, a large part of the tokens are rubbish words or subtokens of a word, which will never occur when using real datasets.

Secondly, although the model is trained on simplified Chinese texts, its vocabulary contains many traditional Chinese characters. This may result in an illusion of users that the model can handle traditional Chinese as well. However, the token are less trained, so the model will never learn the exact meaning of these traditional Chinese characters.

Thirdly, as the vocabulary is fit on Chinese dataset, some of the common Taiwanese Hokkien characters are absent from the vocabulary. For example, 𪜶 ('they; its') is a common character in Taiwanese Hokkien. If the tokenizer cannot handle this word, it will fail to translate many sentences.

TODO: VSCode detected unusual line terminators.

To tackle the first problem, I modified the vocabulary of the original dataset, removed all the rubbish words.

For the second problem, I used a tool to identify all charaters.

For the third problem, I added characters in the dataset.

The resulting vocabulary size is 6995.

## Embedding

Removed

Randomly initialised.

## Dataloader

Since the dataset is very small, the data are directly placed on the GPU memory. However, for larger dataset, the `main` branch provides an on-demand dataloader.

## Training

SGD

## Generation

## Analysis