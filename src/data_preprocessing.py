import os
import fitz  # PyMuPDF
import pandas as pd

# Папка с PDF
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Порог: минимальное количество символов в "работе"
MIN_DOC_LENGTH = 500


def extract_texts_from_pdf(filepath):
    """
    Разбивает PDF-файл на список строк — по одной работе на элемент.
    """
    doc = fitz.open(filepath)
    texts = []

    for page in doc:
        text = page.get_text()
        # Удалим пустые или слишком короткие страницы
        if len(text.strip()) >= MIN_DOC_LENGTH:
            texts.append(text.strip())
    doc.close()
    return texts


def create_dataset():
    """
    Читает два PDF, создаёт общий DataFrame с текстами и метками
    """
    print("[INFO] Загружаем PDF-файлы...")
    student_path = os.path.join(DATA_DIR, 'student_papers.pdf')
    ai_path = os.path.join(DATA_DIR, 'ai_generated_papers.pdf')

    student_texts = extract_texts_from_pdf(student_path)
    ai_texts = extract_texts_from_pdf(ai_path)

    print(f"[INFO] Извлечено: {len(student_texts)} работ от студентов, {len(ai_texts)} от ИИ")

    df = pd.DataFrame({
        'text': student_texts + ai_texts,
        'label': [0] * len(student_texts) + [1] * len(ai_texts)
    })

    return df


def save_dataset(df: pd.DataFrame, out_path='data/dataset.csv'):
    df.to_csv(out_path, index=False)
    print(f"[INFO] Датасет сохранён в {out_path}")


if __name__ == "__main__":
    df = create_dataset()
    save_dataset(df)