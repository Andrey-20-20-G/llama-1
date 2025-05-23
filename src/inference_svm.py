import fitz  # pymupdf
import joblib
import os
import re

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from text_features import lexical_features  # из предыдущего модуля

MODEL_PATH = "models/svm_classifier.joblib"
SCALER_PATH = "models/scaler.joblib"
TFIDF_PATH = "models/tfidf_vocab.joblib"


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = " ".join([page.get_text() for page in doc])
    doc.close()
    return full_text.strip()


def predict_class(text: str, tfidf_vocab: list, model, scaler):
    """
    Возвращает предсказание класса и вероятность или аналог уверенности.
    """

    # 1. TF-IDF
    tfidf_vec = TfidfVectorizer(vocabulary=tfidf_vocab, stop_words='english')
    X_tfidf = tfidf_vec.fit_transform([text])

    # 2. Hand-crafted features
    feats = lexical_features(text)
    feat_array = np.array([[
        feats['n_sentences'],
        feats['n_words'],
        feats['avg_word_length'],
        feats['avg_sent_length'],
        feats['unique_tokens'],
        feats['stopword_ratio']
    ]])
    scaled_feats = scaler.transform(feat_array)

    # 3. Объединяем признаки
    x_combined = hstack([X_tfidf, scaled_feats])

    # 4. Предсказание
    pred_label = model.predict(x_combined)[0]

    # Если у модели есть predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_combined)[0][1]
    else:
        # Используем decision_function и логистическую функцию для оценки вероятности
        score = model.decision_function(x_combined)[0]
        proba = 1 / (1 + np.exp(-abs(score)))  # псевдо-вероятность

    readable_label = convert_proba_to_label(proba)
    return readable_label, proba


def convert_proba_to_label(prob: float) -> str:
    """
    Интерпретирует вероятность в категорию (градиент оценки).
    """
    if prob >= 0.9:
        return "🟥 Написано ИИ"
    elif 0.65 <= prob < 0.9:
        return "🟧 Преимущественно ИИ"
    elif 0.35 <= prob < 0.65:
        return "🟨 Преимущественно человек"
    else:
        return "🟩 Написано человеком"


def run_inference(pdf_path, tfidf_vocab_path=TFIDF_PATH):
    print("[INFO] Загружаем модели...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    tfidf_vocab = joblib.load(tfidf_vocab_path)

    print(f"[INFO] Загружаем файл {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print("[ERROR] PDF файл пустой или не удалось распознать текст.")
        return

    result, proba = predict_class(text, tfidf_vocab, model, scaler)

    print(f"\n✅ Классификация: {result}")
    print(f"🔢 Вероятность, что ИИ: {proba:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Анализ PDF на ИИ-признаки")
    parser.add_argument("--pdf", type=str, required=True, help="Путь к PDF документу")

    args = parser.parse_args()
    run_inference(args.pdf)