import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOPWORDS = set(stopwords.words("english"))
WORD_RE = re.compile(r"[A-Za-z]+")

DISCOURSE_MARKERS = {
    "however", "therefore", "moreover", "thus", "furthermore", "consequently",
    "additionally", "although", "whereas", "meanwhile", "hence", "overall",
    "in contrast", "for example", "for instance", "in addition", "as a result",
}

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def md5_hash(text: str) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()

def clean_text_basic(text: str) -> str:
    text = str(text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\$\$?.+?\$\$?', '', text)
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False

def word_count(text: str) -> int:
    return len(str(text).split())

def save_dataframe_as_png(df: pd.DataFrame, filepath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(df.columns)), max(2.5, 0.5 * len(df) + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

def tokenize_words(text: str):
    return WORD_RE.findall(str(text).lower())

def tokenize_sentences(text: str):
    try:
        sents = sent_tokenize(str(text))
        return [s for s in sents if s.strip()]
    except Exception:
        return [s.strip() for s in re.split(r"[.!?]+", str(text)) if s.strip()]

def split_paragraphs(text: str):
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(text)) if p.strip()]
    return paras if paras else [str(text).strip()]

def safe_mean(vals):
    return float(np.mean(vals)) if len(vals) > 0 else 0.0

def safe_var(vals):
    return float(np.var(vals)) if len(vals) > 1 else 0.0

def stylometric_features(text: str):
    text = str(text)
    words = tokenize_words(text)
    sents = tokenize_sentences(text)
    paras = split_paragraphs(text)

    word_lens = [len(w) for w in words]
    sent_lens = [len(tokenize_words(s)) for s in sents]
    para_lens = [len(tokenize_words(p)) for p in paras]

    total_tokens = max(len(words), 1)
    total_chars = max(len(text), 1)

    punctuation_count = len(re.findall(r"[,:;()\-\[\]{}!?\.]", text))
    stopword_count = sum(1 for w in words if w in STOPWORDS)

    lowered_text = text.lower()
    discourse_count = 0
    for marker in DISCOURSE_MARKERS:
        discourse_count += lowered_text.count(marker)

    return {
        "avg_sentence_length": safe_mean(sent_lens),
        "sentence_length_variance": safe_var(sent_lens),
        "type_token_ratio": len(set(words)) / total_tokens,
        "punctuation_ratio": punctuation_count / total_chars,
        "stopword_ratio": stopword_count / total_tokens,
        "avg_word_length": safe_mean(word_lens),
        "paragraph_length_variance": safe_var(para_lens),
        "discourse_ratio": discourse_count / total_tokens,
    }
