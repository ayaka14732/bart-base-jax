conv_table = {}

with open('STCharacters.txt', encoding='utf-8') as f:
    for line in f:
        simp, trads = line.rstrip('\n').split('\t')
        trads = ''.join(trads.split(' '))
        conv_table[simp] = trads
