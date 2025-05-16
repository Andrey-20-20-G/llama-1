# from text_features import calculate_handcrafted_features, calculate_tfidf_features
# import pandas as pd

# df = pd.read_csv("data/dataset.csv")

# df_feats = calculate_handcrafted_features(df)
# print(df_feats.head())

# tfidf_matrix, feature_names = calculate_tfidf_features(df)
# print(f"TF-IDF shape: {tfidf_matrix.shape}")

from text_features import calculate_handcrafted_features, calculate_tfidf_features
import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv("data/dataset.csv")

# 1. Расчет handcrafted-признаков
df_feats = calculate_handcrafted_features(df)
print("\n[1] Handcrafted-признаки:")
# print(df_feats.head())

pd.set_option('display.max_columns', None)  # Показать все колонки
print(df_feats.head())

# 2. Расчет TF-IDF признаков
tfidf_matrix, feature_names = calculate_tfidf_features(df)
print(f"\n[2] TF-IDF matrix shape: {tfidf_matrix.shape}")

# 3. Примеры векторов TF-IDF
def show_tfidf_examples(matrix, features, n_examples=3, n_features=10):
    """Выводит примеры TF-IDF векторов"""
    print("\n[3] Примеры TF-IDF векторов:")
    for i in range(min(n_examples, matrix.shape[0])):
        # Получаем ненулевые элементы для i-го документа
        row = matrix[i].toarray().flatten()
        non_zero_idx = np.nonzero(row)[0]
        
        # Сортируем по убыванию веса
        sorted_idx = non_zero_idx[np.argsort(-row[non_zero_idx])]
        
        # Выводим топ-N признаков
        print(f"\nДокумент {i+1} (всего {len(non_zero_idx)} ненулевых признаков):")
        for j in sorted_idx[:n_features]:
            print(f"  {features[j]}: {row[j]:.4f}")

# Выводим примеры
show_tfidf_examples(tfidf_matrix, feature_names)

# 4. Сохранение результатов (опционально)
output_path = "data/processed_features.pkl"
pd.to_pickle({
    'handcrafted': df_feats,
    'tfidf_matrix': tfidf_matrix,
    'tfidf_features': feature_names
}, output_path)
print(f"\n[4] Результаты сохранены в {output_path}")