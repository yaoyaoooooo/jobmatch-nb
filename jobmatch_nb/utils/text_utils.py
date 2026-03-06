import re
import jieba

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9+#\.\-_ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chinese_tokenizer(text: str):
    text = normalize_text(text)
    return [w.strip() for w in jieba.lcut(text) if w.strip()]

def join_fields(*args):
    return " ".join([normalize_text(str(x)) for x in args if str(x).strip()])