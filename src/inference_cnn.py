import torch
import torch.nn as nn
import joblib
import fitz
import re
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Пути ===
MODEL_PATH = os.path.join("models", "cnn_model.pt")
VOCAB_PATH = os.path.join("models", "vocab.joblib")
MAX_SEQ_LEN = 300


# === Архитектура CNN (такая же как при обучении) ===
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=128, kernel_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return torch.sigmoid(out).squeeze()


# === Вспомогательные функции ===

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    doc.close()
    return text.strip()


def clean_and_tokenize(text):
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9ёЁ\s]", "", text.lower())
    return text.split()


def text_to_tensor(text, vocab, max_len=MAX_SEQ_LEN):
    tokens = clean_and_tokenize(text)[:max_len]
    ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]
    padded = ids + [0] * (max_len - len(ids))
    return torch.tensor([padded], dtype=torch.long).to(DEVICE)


def interpret_probability(prob):
    if prob >= 0.9:
        return "🟥 Написано ИИ"
    elif prob >= 0.65:
        return "🟧 Преимущественно ИИ"
    elif prob >= 0.35:
        return "🟨 Преимущественно человек"
    else:
        return "🟩 Написано человеком"


# === Основная функция ===

def run_inference(pdf_path):
    print("[INFO] Загрузка модели CNN и словаря...")
    vocab = joblib.load(VOCAB_PATH)

    model = CNNClassifier(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"[INFO] Чтение текста из файла: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("[ERROR] PDF-файл пустой или текст не найден.")
        return

    input_tensor = text_to_tensor(text, vocab)

    with torch.no_grad():
        proba = model(input_tensor).item()

    label = interpret_probability(proba)

    print(f"\n✅ Классификация: {label}")
    print(f"🔢 Вероятность, что ИИ: {proba:.4f}")


# === Запуск скрипта из командной строки ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Инференс CNN модели (PyTorch)")
    parser.add_argument("--pdf", required=True, help="Путь к PDF файлу")
    args = parser.parse_args()

    run_inference(args.pdf)