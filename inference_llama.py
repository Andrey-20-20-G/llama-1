import os
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import tempfile

MODEL_PATH = "models/llama3-finetuned"  # Путь к дообученной LLaMA-3 или OpenChat


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает весь текст из PDF файла
    """
    print(f"[PDF] Извлекаем текст из файла: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()
    return full_text.strip()


def build_prompt(text: str) -> list:
    """
    Генерирует chat-подобный prompt в формате обучения
    """
    return [
        {
            "role": "user",
            "content": f"Определи, кто написал следующий текст:\n\n{text}\n\nОтвет:"
        }
    ]


# def load_model(model_path: str):
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         device_map="auto"
#     )
#     return tokenizer, model

# def load_model(model_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

#     # Создаем временную директорию для offload (или укажи свою)
#     offload_dir = tempfile.mkdtemp()

#     print(f"[⚙️] Используется offload_dir: {offload_dir}")

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         offload_folder=offload_dir,
#         offload_state_dict=True
#     )

#     return tokenizer, model

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )

    return tokenizer, model


def predict_from_pdf(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)

    if len(text) < 200:
        raise ValueError("❌ Слишком короткий текст для анализа.")

    print(f"[INFO] Длина статьи: {len(text)} символов")

    tokenizer, model = load_model(MODEL_PATH)

    messages = build_prompt(text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    print("[🧠] Генерируем ответ...")

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.0
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Извлекаем только часть после prompt
    generated_part = decoded.split(messages[0]["content"])[-1].strip()

    print("\n📜 Вопрос к модели:")
    print(messages[0]["content"])
    print("\n🤖 Ответ модели:")
    print(generated_part)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLama-инференс по PDF статье")
    parser.add_argument("--pdf", type=str, required=True, help="Путь к статье формата PDF")
    args = parser.parse_args()

    if not os.path.exists(args.pdf) or not args.pdf.endswith(".pdf"):
        print("❌ Укажите правильный путь к PDF файлу.")
    else:
        predict_from_pdf(args.pdf)