import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
import torch.nn.functional as F

from collections import Counter
import re

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tokenize(text):
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', '', text.lower())
    return text.split()


def build_vocab(texts, max_size=10000, min_freq=1):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, freq in counter.items():
        if freq >= min_freq and len(vocab) < max_size:
            vocab[token] = len(vocab)
    return vocab


def text_to_sequence(text, vocab):
    return [vocab.get(tok, vocab['<UNK>']) for tok in tokenize(text)]

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.sequences = [torch.tensor(text_to_sequence(text, vocab)) for text in texts]
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_batch(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return texts_padded.to(DEVICE), labels.to(DEVICE)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        logits = self.fc(h_n[-1])
        return torch.sigmoid(logits).squeeze()


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=128, kernel_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return torch.sigmoid(logits).squeeze()
    
def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X)
                all_preds.extend((preds > 0.5).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        print(classification_report(all_labels, all_preds, digits=4))
        
def train_models():
    df = pd.read_csv("data/dataset.csv")
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()

    vocab = build_vocab(texts)
    joblib.dump(vocab, os.path.join(MODEL_DIR, 'vocab.joblib'))

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_ds = TextDataset(X_train_texts, y_train, vocab)
    test_ds = TextDataset(X_test_texts, y_test, vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_batch)

    # CNN
    print("\n[INFO] Обучаем CNN...")
    cnn = CNNClassifier(vocab_size=len(vocab))
    train_model(cnn, train_loader, test_loader)
    torch.save(cnn.state_dict(), os.path.join(MODEL_DIR, "cnn_model.pt"))

    # LSTM
    print("\n[INFO] Обучаем LSTM (RNN)...")
    lstm = LSTMClassifier(vocab_size=len(vocab))
    train_model(lstm, train_loader, test_loader)
    torch.save(lstm.state_dict(), os.path.join(MODEL_DIR, "rnn_model.pt"))

    print("[✔] Обучение завершено.")
    
if __name__ == "__main__":
    train_models()