import pandas as pd
import numpy as np
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

from collections import Counter
import math

# Убедись, что токены загружены
nltk.download("punkt")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words('russian'))
russian_stop_words = stopwords.words('russian')

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

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
    """
    Считает количество абзацев в тексте (разделены двумя и более переносами строки).
    """
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    return len([p for p in paragraphs if p.strip() != ''])

def has_numbered_lists(text):
    """
    Проверяет наличие нумерованных списков в тексте (например, '1. ... 2. ...').
    Возвращает:
      - 0, если списков нет
      - 1, если есть простые нумерованные списки
      - 2, если есть многоуровневые списки (например, '1.1. ...')
    """
    # Простые нумерованные списки (1., 2., ...)
    simple_list = re.search(r'^\d+\.\s+.+$', text, re.MULTILINE)
    
    # Многоуровневые списки (1.1., 1.2., ...)
    nested_list = re.search(r'^\d+\.\d+\.\s+.+$', text, re.MULTILINE)
    
    if nested_list:
        return 2
    elif simple_list:
        return 1
    else:
        return 0
    
def has_bullet_lists(text):
    """
    Проверяет наличие маркированных списков (например, '- ...', '* ...').
    """
    return int(re.search(r'^(\s*[-*•])\s+.+$', text, re.MULTILINE) is not None)

def list_density(text):
    """
    Доля строк, являющихся элементами списка (нумерованного или маркированного).
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    list_items = sum(1 for line in lines if re.match(r'^(\d+\.|\s*[-*•])\s+', line))
    return list_items / len(lines) if lines else 0

def calculate_perplexity(text, n=2):
    """
    Вычисляет перплексию текста на основе N-граммной модели.
    :param text: входной текст
    :param n: размер N-граммы (по умолчанию биграммы)
    :return: значение перплексии
    """
    words = word_tokenize(text.lower())
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Считаем частоты N-грамм
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    
    # Вероятности и энтропия
    entropy = 0.0
    for ngram in ngram_counts:
        prob = ngram_counts[ngram] / total_ngrams
        entropy -= prob * math.log(prob, 2)
    
    perplexity = 2 ** entropy
    return perplexity

def calculate_burstiness(text):
    """
    Вычисляет бёрстиность текста.
    :param text: входной текст
    :return: значение бёрстиности от -1 до 1
    """
    words = word_tokenize(text.lower())
    word_positions = {}
    
    # Запоминаем позиции каждого слова
    for i, word in enumerate(words):
        if word not in word_positions:
            word_positions[word] = []
        word_positions[word].append(i)
    
    # Считаем интервалы между повторениями слов
    intervals = []
    for word, positions in word_positions.items():
        if len(positions) > 1:
            for i in range(1, len(positions)):
                intervals.append(positions[i] - positions[i-1])
    
    if not intervals:
        return 0.0  # Нет повторяющихся слов
    
    # Формула бёрстиности
    mu = np.mean(intervals)
    sigma = np.std(intervals)
    burstiness = (sigma - mu) / (sigma + mu) if (sigma + mu) != 0 else 0.0
    return burstiness


def lexical_features(text):
    """
    Простейшие статистики по тексту.
    """
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
        "has_bullet_lists" : has_bullet_lists(text),
        "list_density" : list_density(text),
        
        "list_item_density": len(re.findall(r'^\d+\.\s+', text, re.MULTILINE)) / len(sentences) if sentences else 0,
    }

    return features


def calculate_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет набор функций ко всем строкам
    """
    print("[INFO] Считаем метрики текста...")

    all_features = []
    for i, text in enumerate(df['text']):
        feats = lexical_features(text)
        feats["perplexity"] = calculate_perplexity(text)
        feats["burstiness"] = calculate_burstiness(text)
        all_features.append(feats)
        if (i + 1) % 100 == 0:
            print(f"   Обработано {i+1} текстов...")

    feats_df = pd.DataFrame(all_features)
    return pd.concat([df.reset_index(drop=True), feats_df], axis=1)


def calculate_tfidf_features(df: pd.DataFrame, max_features=200):
    """
    Генерирует TF-IDF признаки и возвращает матрицу
    """
    print("[INFO] Считаем TF-IDF...")
    tfidf = TfidfVectorizer(max_features=max_features, stop_words=russian_stop_words)

    X_tfidf = tfidf.fit_transform(df['text'])
    return X_tfidf, tfidf.get_feature_names_out()