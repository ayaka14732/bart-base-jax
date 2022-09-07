from .is_trad_ch import is_trad_ch
from .all_chars_in_data import all_chars_in_data

def is_symbol(c):
    return c in '👍🔥😂😎ぁあぃいうぇえおかきくけこさしすせそたちっつてとなにぬねのはひふへほまみむめもゃやゅゆょよらりるれろわをん゜ゝァアィイゥウェエォオカキクケコサシスセソタチッツテトナニヌネノハヒフヘホマミムメモャヤュユョヨラリルレロワヲンヶ・ーヽㄅㄆㄇㄉㄋㄌㄍㄎㄏㄒㄚㄛㄞㄟㄢㄤㄥㄧㄨㆍ㈦㊣㎡ابةتدرسعلمنهوي۩กงนมยรอาเ๑་ღᄀᄁᄂᄃᄅᄆᄇᄈᄉᄋᄌᄎᄏᄐᄑ하ᅢᅣᅥᅦᅧᅨᅩᅪᅬᅭᅮᅯᅲᅳᅴᅵᆨᆫᆯᆷᆸᆺᆻᆼᗜ─━│┃┅┆┊┌└├┣═║╚╞╠╭╮╯╰╱╳▂▃▅▇█▉▋▌▍▎■□▪▫▬▲△▶►▼▽◆◇○◎●◕◠◢◤☀★☆☕☞☺☼♀♂♠♡♣♥♦♪♫♬✈✔✕✖✦✨✪✰✿❀❤➜➤⦿'

def is_special_token(c):
    return c in [
        '[PAD]',
        '[UNK]',
        '[CLS]',
        '[SEP]',
        '[MASK]',
    ]

def should_remove(c):
    return not is_special_token(c) and (len(c) > 1 or is_trad_ch(c) or is_symbol(c)) and c not in all_chars_in_data
