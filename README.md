# BART Base Cantonese

This is the Cantonese model of BART base. It is obtained by a second-stage pre-training on the [LIHKG dataset](https://github.com/ayaka14732/lihkg-scraper) based on the [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) model.

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Usage

```python
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')
model = BartForConditionalGeneration.from_pretrained('Ayaka/bart-base-cantonese')
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
output = text2text_generator('聽日就要返香港，我激動到[MASK]唔着', max_length=50, do_sample=False)
print(output[0]['generated_text'].replace(' ', ''))
# output: 聽日就要返香港，我激動到瞓唔着
```
