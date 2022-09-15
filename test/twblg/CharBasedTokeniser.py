from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from os.path import join
import tempfile

from lib.twblg.CharBasedTokeniser import CharBasedTokeniser

vocab = [
    '[PAD]',
    '[UNK]',
    '[BOS]',
    '[EOS]',
    '[MSK]',
    '五',
    '六',
    '七',
    '八',
    '九',
]

tokeniser = CharBasedTokeniser(vocab=vocab)
with tempfile.TemporaryDirectory() as tmpdirname:
    tokeniser.save_vocabulary(save_directory=tmpdirname, filename_prefix='prefix_')
    del tokeniser
    vocab_file = join(tmpdirname, 'prefix_vocab.txt')
    tokeniser = CharBasedTokeniser(vocab=vocab_file)

inputs = tokeniser(['五六', '七八九十'], return_tensors='np', max_length=8, padding='max_length', truncation=True)
inputs.input_ids[0].tolist()[:4] == [2, 5, 6, 3]
inputs.input_ids[1].tolist()[:6] == [2, 7, 8, 9, 1, 3]
inputs.attention_mask[0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
inputs.attention_mask[1].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
