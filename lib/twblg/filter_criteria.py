from .certain_trad import certain_trad
from .should_add import should_add

def is_symbol(c):
    return c in '👍🔥😂😎ㆍ㈦㊣㎡ابةتدرسعلمنهوي۩กงนมยรอาเ๑་ღᄀᄁᄂᄃᄅᄆᄇᄈᄉᄋᄌᄎᄏᄐᄑ하ᅢᅣᅥᅦᅧᅨᅩᅪᅬᅭᅮᅯᅲᅳᅴᅵᆨᆫᆯᆷᆸᆺᆻᆼᗜ─━│┃┅┆┊┌└├┣═║╚╞╠╭╮╯╰╱╳▂▃▅▇█▉▋▌▍▎■□▪▫▬▲△▶►▼▽◆◇○◎●◕◠◢◤☀★☆☕☞☺☼♀♂♠♡♣♥♦♪♫♬✈✔✕✖✦✨✪✰✿❀❤➜➤⦿\u2028'

def is_special_token(c):
    return c in [
        '[PAD]',
        '[UNK]',
        '[CLS]',
        '[SEP]',
        '[MASK]',
    ]

def should_remove(c):
    return not is_special_token(c) and (len(c) > 1 or certain_trad(c) or is_symbol(c)) and c not in should_add
