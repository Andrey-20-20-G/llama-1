import torch
import torch.nn as nn
import joblib
import fitz
import re
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Настройки ===
MODEL_PATH = os.path.join("models", "rnn_model.pt")
VOCAB_PATH = os.path.join("models", "vocab.joblib")
MAX_SEQ_LEN = 300


# === Модель RNN такая же, как при обучении ===
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden_state, _) = self.lstm(x)
        out = self.fc(hidden_state[-1])
        return torch.sigmoid(out).squeeze()


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    doc.close()
    return text.strip()


def clean_and_tokenize(text):
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9ёЁ\s]", "", text.lower())
    return text.split()


def text_to_tensor(text, vocab, max_len=MAX_SEQ_LEN):
    tokens = clean_and_tokenize(text)
    tokens = tokens[:max_len]
    ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens]

    # Padding
    padded = ids + [0] * (max_len - len(ids))
    tensor_input = torch.tensor([padded], dtype=torch.long)
    return tensor_input.to(DEVICE)


def interpret_probability(prob):
    if prob >= 0.9:
        return "🟥 Написано ИИ"
    elif prob >= 0.65:
        return "🟧 Преимущественно ИИ"
    elif prob >= 0.35:
        return "🟨 Преимущественно человек"
    else:
        return "🟩 Написано человеком"


def run_inference(pdf_path):
    print("[INFO] Загрузка модели и словаря...")
    vocab = joblib.load(VOCAB_PATH)
    model = LSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"[INFO] Чтение PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("[ERROR] Файл PDF пуст или не содержит текста.")
        return

    # Преобразуем текст
    input_tensor = text_to_tensor(text, vocab)

    with torch.no_grad():
        proba = model(input_tensor).item()

    label = interpret_probability(proba)

    print(f"\n✅ Классификация: {label}")
    print(f"🔢 Вероятность ИИ: {proba:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Инференс RNN модели на PyTorch")
    parser.add_argument("--pdf", required=True, help="Путь к PDF файлу")
    args = parser.parse_args()

    run_inference(args.pdf)
    