# Machine Translation Model from Mandarin to Taiwanese Hokkien

This project is an implementation of the bart-base model for machine translation from Mandarin to Taiwanese Hokkien.

This project is trained on the compute cluster provided by NUS SoC.

The project is based on the `main` branch of this repository, [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax), a JAX implementation of the bart-base model, supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Environment Setup

The scripts are designed to be run on a single machine with one GPU.

```sh
python3.10 -m venv ./venv
. ./venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[tpu]==0.3.17" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
```

## Dataset

The dataset is provided in `lib/twblg/data.tsv`. It is a parallel corpus that consists of 8366 sentences. The sentences are split into train, dev and test sets with a ratio of 8:1:1. However, at present, the dev set is not utilised during training.

The source of the dataset is [g0v/moedict-data-twblg](https://github.com/g0v/moedict-data-twblg/blob/master/uni/%E4%BE%8B%E5%8F%A5.csv). The advantage of this dataset is that the sentences are derived from the dictionary. Therefore, it is more effective for the model to learn the distinctive vocabulary of Taiwanese Hokkien.

## Data Preprocessing

As I am going to use [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) as my base model, the dataset should be written in simplified Chinese. Therefore, I converted all the texts in the dataset from traditional Chinese to simplified Chinese using [StarCC](https://github.com/StarCC0/starcc-py).

Due to the one-to-many problem, the accuracy is not always guaranteed when converting from simplified Chinese to traditional Chinese. To tackle this, I modify StarCC's conversion algorithm from two aspects:

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

The resulting vocabulary size is 7697.

## Embedding

Removed

Randomly initialised.

## Dataloader

Since the dataset is very small, the data are directly placed on the GPU memory. However, for larger dataset, the `main` branch provides an on-demand dataloader.

## Training

SGD

## Generation

## Analysis
