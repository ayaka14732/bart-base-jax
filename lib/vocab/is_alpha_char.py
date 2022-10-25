# alpha character: characters with multiple simplified forms and one of them is the same as the original one

alpha_chars = set()

with open('TSCharacters.txt', encoding='utf-8') as f:
    for line in f:
        trad, simps = line.rstrip('\n').split('\t')
        simps = simps.split(' ')
        if trad in simps:
            alpha_chars.add(trad)

def is_alpha_char(c: str) -> bool:
    return c in alpha_chars
