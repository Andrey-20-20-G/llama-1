import pandas as pd
import numpy as np
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Убедись, что токены загружены
nltk.download("punkt")
nltk.download("stopwords")


STOP_WORDS = set(stopwords.words('russian'))


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def lexical_features(text):
    """
    Простейшие статистики по тексту.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    words_clean = [w.lower() for w in words if w.isalpha()]
    words_no_stop = [w for w in words_clean if w not in STOP_WORDS]
    sent_lens = [len(sent.split()) for sent in sentences]

    features = {
        "n_sentences": len(sentences),
        "n_words": len(words),
        "avg_word_length": np.mean([len(w) for w in words_clean]) if words_clean else 0,
        "avg_sent_length": np.mean(sent_lens) if sent_lens else 0,
        "unique_tokens": len(set(words_clean)),
        "stopword_ratio": len([w for w in words_clean if w in STOP_WORDS]) / len(words_clean) if words_clean else 0,
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
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')

    X_tfidf = tfidf.fit_transform(df['text'])
    return X_tfidf, tfidf.get_feature_names_out()