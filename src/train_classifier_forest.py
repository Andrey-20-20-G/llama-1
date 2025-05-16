import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
import os

from text_features import calculate_handcrafted_features, calculate_tfidf_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_classifier.joblib')


def train_models():
    print("[INFO] Загружаем датасет...")
    df = pd.read_csv("data/dataset.csv")

    # 1. Вычисляем признаки
    df_feats = calculate_handcrafted_features(df)
    tfidf_matrix, tfidf_features = calculate_tfidf_features(df)

    print("[INFO] Объединяем фичи...")

    # 2. Получаем числовые признаки
    numeric_feats = df_feats[['n_sentences', 'n_words', 'avg_word_length',
                              'avg_sent_length', 'unique_tokens', 'stopword_ratio']]

    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(numeric_feats)

    # 3. Объединяем TF-IDF + другие признаки
    X_combined = hstack([tfidf_matrix, scaled_feats])
    y = df_feats['label'].values

    # 4. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )

    # 5. Обучение моделей
    print("[INFO] Обучение модели...")
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    # 6. Предсказание
    y_pred = clf.predict(X_test)

    # 7. Метрики
    print("\n📊 Classification report:")
    print(classification_report(y_test, y_pred))

    print("📉 Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Сохраняем модель
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(tfidf_features, 'models/tfidf_vocab.joblib')  # Добавить
    print(f"[INFO] Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    train_models()
    