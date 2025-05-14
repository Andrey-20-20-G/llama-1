import pandas as pd
import json
import os

INPUT_PATH = "data/dataset.csv"
OUTPUT_PATH = "data/llama_train.json"

def format_text(row):
    text = row['text']
    label = row['label']
    
    return {
        "messages": [
            {"role": "user", "content": f"Определи: кто написал этот текст?\n\n{text}"},
            {"role": "assistant", "content": "ИИ" if label == 1 else "Человек"}
        ]
    }

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    formatted_data = [format_text(row) for _, row in df.iterrows()]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Данные сохранены в {OUTPUT_PATH}")