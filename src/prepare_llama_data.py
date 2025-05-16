import pandas as pd
import json
import os
import string
import re
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
import nltk

# Загрузка ресурсов NLTK
nltk.download("punkt")
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words('russian'))


# === ФУНКЦИИ ДЛЯ ПРИЗНАКОВ ===

def count_syllables(word):
    vowels = "аеёиоуыэюя"
    return sum(1 for char in word.lower() if char in vowels)

def calculate_flesch_reading_ease(text):
    words = word_tokenize(text, language='russian')
    sentences = sent_tokenize(text, language='russian')
    avg_words_per_sent = len(words) / len(sentences) if sentences else 0
    avg_syllables_per_word = sum(count_syllables(w) for w in words) / len(words) if words else 0
    return 206.835 - 1.015 * avg_words_per_sent - 84.6 * avg_syllables_per_word

def calculate_yules_k(words):
    word_counts = Counter(words)
    m1 = sum(word_counts.values())
    m2 = sum(cnt * cnt for cnt in word_counts.values())
    return (m1 * m1) / (m2 - m1) if (m2 - m1) != 0 else 0

def calculate_honore_statistic(words):
    v1 = len([w for w, cnt in Counter(words).items() if cnt == 1])
    n = len(words)
    return 100 * np.log(n) / (1 - v1 / n) if n > 0 and v1 != n else 0

def count_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    return len([p for p in paragraphs if p.strip() != ''])

def has_numbered_lists(text):
    simple_list = re.search(r'^\d+\.\s+.+$', text, re.MULTILINE)
    nested_list = re.search(r'^\d+\.\d+\.\s+.+$', text, re.MULTILINE)
    if nested_list:
        return 2
    elif simple_list:
        return 1
    else:
        return 0

def has_bullet_lists(text):
    return int(re.search(r'^(\s*[-*•])\s+.+$', text, re.MULTILINE) is not None)

def list_density(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    list_items = sum(1 for line in lines if re.match(r'^(\d+\.|\s*[-*•])\s+', line))
    return list_items / len(lines) if lines else 0

def calculate_perplexity(text, n=2):
    words = word_tokenize(text.lower())
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    entropy = 0.0
    for ngram in ngram_counts:
        prob = ngram_counts[ngram] / total_ngrams
        entropy -= prob * math.log(prob, 2)
    perplexity = 2 ** entropy
    return perplexity

def calculate_burstiness(text):
    words = word_tokenize(text.lower())
    word_positions = {}
    for i, word in enumerate(words):
        word_positions.setdefault(word, []).append(i)
    intervals = []
    for positions in word_positions.values():
        if len(positions) > 1:
            for i in range(1, len(positions)):
                intervals.append(positions[i] - positions[i-1])
    if not intervals:
        return 0.0
    mu = np.mean(intervals)
    sigma = np.std(intervals)
    return (sigma - mu) / (sigma + mu) if (sigma + mu) != 0 else 0.0

def lexical_features(text):
    sentences = sent_tokenize(text, language='russian')
    words = word_tokenize(text, language='russian')
    words_clean = [w.lower() for w in words if w.isalpha()]
    words_no_stop = [w for w in words_clean if w not in STOP_WORDS]
    sent_lens = [len(sent.split()) for sent in sentences]
    word_counts = Counter(words_clean)

    features = {
        "n_sentences": len(sentences),
        "n_words": len(words),
        "n_unique_words": len(set(words_clean)),
        "n_chars": sum(len(w) for w in words),
        "n_syllables": sum(count_syllables(w) for w in words_clean),
        "avg_word_length": np.mean([len(w) for w in words_clean]) if words_clean else 0,
        "avg_sent_length": np.mean(sent_lens) if sent_lens else 0,
        "avg_syllables_per_word": np.mean([count_syllables(w) for w in words_clean]) if words_clean else 0,
        "stopword_ratio": len([w for w in words_clean if w in STOP_WORDS]) / len(words_clean) if words_clean else 0,
        "lexical_diversity": len(set(words_clean)) / len(words_clean) if words_clean else 0,
        "flesch_reading_ease": calculate_flesch_reading_ease(text),
        "punctuation_ratio": sum(1 for w in words if not w.isalpha()) / len(words) if words else 0,
        "uppercase_ratio": sum(1 for w in words if w.isupper()) / len(words) if words else 0,
        "hapax_legomena_ratio": len([w for w, cnt in word_counts.items() if cnt == 1]) / len(words_clean) if words_clean else 0,
        "yules_k": calculate_yules_k(words_clean),
        "honore_statistic": calculate_honore_statistic(words_clean),
        "unique_tokens": len(set(words_clean)),
        "n_paragraphs": count_paragraphs(text),
        "has_numbered_lists": has_numbered_lists(text),
        "has_bullet_lists": has_bullet_lists(text),
        "list_density": list_density(text),
        "perplexity": calculate_perplexity(text),
        "burstiness": calculate_burstiness(text)
    }

    return features


# === ОСНОВНАЯ ЛОГИКА ФОРМАТИРОВАНИЯ ===

INPUT_PATH = "data/dataset.csv"
OUTPUT_PATH = "data/llama_train.json"

def extract_features_text(text):
    features = lexical_features(text)
    # Формируем строку вида: "[FEATURES] key1: val1; key2: val2 ..."
    feature_str = '; '.join(f"{k}: {round(v, 4) if isinstance(v, float) else v}" for k, v in features.items())
    return f"[FEATURES] {feature_str}"

def format_text(row):
    text = row['text']
    label = row['label']
    features_str = extract_features_text(text)

    return {
        "messages": [
            {"role": "system", "content": features_str},
            {"role": "user", "content": f"Определи: кто написал этот текст?\n\n{text}"},
            {"role": "assistant", "content": "ИИ" if label == 1 else "Человек"}
        ]
    }

# === ЗАПУСК ===

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    formatted_data = [format_text(row) for _, row in df.iterrows()]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Данные сохранены в {OUTPUT_PATH}")