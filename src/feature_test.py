from text_features import calculate_handcrafted_features, calculate_tfidf_features
import pandas as pd

df = pd.read_csv("data/dataset.csv")

df_feats = calculate_handcrafted_features(df)
print(df_feats.head())

tfidf_matrix, feature_names = calculate_tfidf_features(df)
print(f"TF-IDF shape: {tfidf_matrix.shape}")