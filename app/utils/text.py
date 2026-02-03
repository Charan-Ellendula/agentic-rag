import re

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")

def clean_text(text: str) -> str:
    # Normalize spaces
    text = text.replace("\u00a0", " ")

    # Remove tokenizer special tokens like <|endoftext|>, <|fim_prefix|>, etc.
    text = _SPECIAL_TOKEN_RE.sub(" ", text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

