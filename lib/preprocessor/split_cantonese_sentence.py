import regex as re

pattern = re.compile(r'\b|(?<=[\p{Unified_Ideograph}\u3006\u3007])|(?=[\p{Unified_Ideograph}\u3006\u3007])')

def split_cantonese_sentence(s: str) -> list[str]:
    return [part for part in pattern.split(s) if part]
