from os.path import abspath, dirname, join

here = dirname(abspath(__file__))

def load_dummy() -> list[str]:
    english_sentences = {}

    with open(join(here, 'tatoeba-uyghur-english-2022-08-28.tsv')) as f:
        for line in f:
            _, _, _, english = line.rstrip('\n').split('\t')
            english_sentences[english] = None  # remove duplicates while maintaining order

    return list(english_sentences)
