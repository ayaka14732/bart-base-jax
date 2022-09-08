class CharBasedTokeniser:
    def __init__(self, vocab: list[str]) -> None:
        self.id2ch = vocab
        self.ch2id = {c: i for i, c in enumerate(vocab)}

        self.vocab_size = len(vocab)

        self.pad_token = self.ch2id['[PAD]']
        self.unk_token = self.ch2id['[UNK]']
        self.cls_token = self.ch2id['[CLS]']
        self.sep_token = self.ch2id['[SEP]']
        self.msk_token = self.ch2id['[MASK]']

    @classmethod
    def from_vocab_file(cls, path: str):
        vocab = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                c = line.rstrip('\n')
                vocab.append(c)
        return cls(vocab)

    def tokenise_sentence(self, s: str) -> list[str]:
        return [
            self.cls_token,
            *(self.ch2id.get(c, self.unk_token) for c in s),
            self.sep_token,
        ]

    def detokenise_sentence(self, l: list) -> list[str]:
        while l and l[-1] == self.pad_token:
            l.pop()
        if l and l[0] == self.cls_token:
            l = l[1:]
        if l and l[-1] == self.sep_token:
            l.pop()
        return ''.join(self.id2ch[i] for i in l)

    def __call__(self, s: str):
        pass
