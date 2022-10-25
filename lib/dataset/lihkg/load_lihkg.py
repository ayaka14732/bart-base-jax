from os.path import expanduser

def load_lihkg() -> list[str]:
    filename = expanduser('~/lihkg-1-2850000-processed-dedup.csv')
    sentences = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            sentences.append(line.rstrip('\n'))
    print(f'INFO: Loaded {len(sentences)} sentences.')
    return sentences
