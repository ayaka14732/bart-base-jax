import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import jax.numpy as np
import sys

from lib.param_utils.load_params import load_params
from lib.en_kfw_nmt.fwd_transformer_encoder_part import fwd_transformer_encoder_part

from transformers import MarianTokenizer, FlaxMarianMTModel

params = load_params('likely-dawn-4-0.dat') #sys.argv[1])
params = jax.tree_map(np.asarray, params)

model = FlaxMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')

# generate

sentences = [
    'anaemia',
    'Get out!',
    'The sky is blue.',
    'Save the children!',
    'enter the spotlight',
    'Are you feeling unwell?',
    'How long have you been waiting?',
    'There are many bubbles in soda water.',
    'He feels deeply melancholic for his past.',
    'She prepared some mooncakes for me to eat.',
    'This gathering only allows adults to join.',
    "Do you know it's illegal to recruit triad members?",
    "Today I'd like to share some tips about making a cake.",
    'An annual fee is required if one wants to use the bank counter service.',
    'You guys have no discipline, how can you be part of the disciplined services?',
    'We need to offer young people drifting into crime an alternative set of values.',
    'A tiger is put with equal probability behind one of two doors, while treasure is put behind the other one.',
    "Recently he's been working so hard to get rid of his nerdy way of living, now he looks so different than before.",
    'Running the marathon race in such a hot day, I almost collapsed as I arrived the destination.',
    'He says flowery and pretty words, no one knows when his praise is real and when is not.',
    'After 10 years of investigation, the police can finally figure out the man behind this murder case.',
    "She has vowed for many times that she would slim down and thus never eat anything fatty, but she still has a fried chicken thigh every week. She can't keep her commitment.",
    'Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.',
    # sentences in the training dataset
    'Adults should protect children so as to avoid them being sexually abused.',
    'Clerks working on a construction site are also construction site workers, engineers are also construction site workers.',
    "A taxi opposite me didn't use its signal light, and when it was driving on my right front side it suddenly turned in toward my left. Luckily I was able to stop in time. Otherwise I would have been hit hard!",
]

inputs = tokenizer(sentences, return_tensors='jax', padding=True)
generate_ids = model.generate(**inputs, num_beams=5, max_length=128, params=params).sequences

decoded_sentences = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for sentence in decoded_sentences:
    sentence = sentence.replace(' ', '')
    print(sentence)
