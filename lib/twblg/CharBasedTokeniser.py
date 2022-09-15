from os import makedirs
from os.path import join
from transformers import AddedToken, PreTrainedTokenizer
from typing import Union, Optional

class CharBasedTokeniser(PreTrainedTokenizer):

    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab: Union[list[str], str]) -> None:
        if isinstance(vocab, str):
            with open(vocab, encoding='utf-8') as f:
                vocab = [line.rstrip('\n') for line in f]

        super().__init__(
            pad_token=AddedToken('[PAD]'),
            unk_token=AddedToken('[UNK]'),
            bos_token=AddedToken('[BOS]'),
            eos_token=AddedToken('[EOS]'),
            mask_token=AddedToken('[MSK]'),
        )

        assert vocab[0] == '[PAD]'
        assert vocab[1] == '[UNK]'
        assert vocab[2] == '[BOS]'
        assert vocab[3] == '[EOS]'
        assert vocab[4] == '[MSK]'

        self.special_tokens_encoder = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
            self.mask_token: 4,
        }
        self._num_special_tokens = len(self.special_tokens_encoder)
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens_encoder.items()}

        self.id2ch = vocab
        self.ch2id = {c: i for i, c in enumerate(vocab)}

        assert self.pad_token_id == 0
        assert self.unk_token_id == 1
        assert self.bos_token_id == 2
        assert self.eos_token_id == 3
        assert self.mask_token_id == 4

    @property
    def vocab_size(self):
        return len(self.id2ch)

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1) -> list[int]:
        assert token_ids_1 is None
        return [self.bos_token_id, *token_ids_0, self.eos_token_id]

    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token):
        unk_token_id = self.ch2id[self.unk_token]
        return self.ch2id.get(token, unk_token_id)

    def _convert_id_to_token(self, index):
        return self.id2ch[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def save_vocabulary(self, save_directory: Optional[str]=None, filename_prefix: Optional[str]=None) -> tuple[str]:
        filename = 'vocab.txt'
        if filename_prefix is not None:
            filename = filename_prefix + filename
        if save_directory is not None:
            makedirs(save_directory, exist_ok=True)
            filename = join(save_directory, filename)

        with open(filename, 'w', encoding='utf-8') as f:
            for w in self.id2ch:
                print(w, file=f)
