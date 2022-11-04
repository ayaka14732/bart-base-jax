# English-Cantonese Translation Model

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Develop

```sh
# Clone repo and datasets
git clone https://github.com/ayaka14732/bart-base-jax.git
git clone https://github.com/CanCLID/abc-cantonese-parallel-corpus.git
git clone https://github.com/CanCLID/wordshk-parallel-corpus.git

# 1st-stage fine-tuning
cd bart-base-jax
git checkout en-kfw-nmt
python 1_convert_bart_params.py
python 2_finetune.py

# 2nd-stage fine-tuning
git checkout en-kfw-nmt-2nd-stage
python 2_finetune.py

# Generate results
python 3_predict.py
python compute_bleu.py results-bart.txt

# Compare with Bing Translator
export TRANSLATE_KEY=...
export ENDPOINT=...
export LOCATION=...
python translate_bing.py
python compute_bleu.py results-bing.txt
```
