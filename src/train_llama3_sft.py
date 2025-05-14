# import os
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# from transformers import Trainer
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import torch
# from transformers.integrations import TensorBoardCallback

# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # или micro версию

# def main():
#     os.environ["WANDB_DISABLED"] = "true"  # отключаем wandb
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     dataset = load_dataset("json", data_files="data/llama_train.json", split="train")

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         torch_dtype=torch.float16,
#         device_map="auto",
#     )

#     model = prepare_model_for_kbit_training(model)

#     peft_config = LoraConfig(
#         r=8,
#         lora_alpha=16,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#         lora_dropout=0.1,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )

#     model = get_peft_model(model, peft_config)

#     def tokenize(example):
#         return tokenizer.apply_chat_template(example["messages"], return_tensors="pt", truncation=True, max_length=1024)

#     dataset = dataset.map(tokenize)

#     training_args = TrainingArguments(
#         output_dir="models/llama3-finetuned",
#         num_train_epochs=2,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         learning_rate=2e-4,
#         logging_dir="outputs/logs",
#         logging_steps=10,
#         save_strategy="epoch",
#         fp16=True,
#         report_to=[],
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         tokenizer=tokenizer,
#         callbacks=[TensorBoardCallback()]
#     )

#     print("🔧 Начинаем обучение...")
#     trainer.train()

#     print("✅ Сохраняем модель...")
#     trainer.save_model("models/llama3-finetuned")

# if __name__ == "__main__":
#     main()

# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
# from peft import LoraConfig
# from trl import SFTTrainer
# import torch

# # === Модель LLaMA-3 ===
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # открытые альтернативы: TinyLlama, OpenChat

# # === Датасет ===
# DATA_PATH = "data/llama_train.json"

# # === Загружаем jsonl
# dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# # === Сжимаем вес модели для экономии (LoRA + 4bit)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# # === Загружаем токенизатор + модель
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, padding_side="right")
# tokenizer.pad_token = tokenizer.eos_token

# # === Настраиваем лору
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# # === Модель
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config,
#     device_map="auto"
# )

# # === Аргументы обучения
# training_args = TrainingArguments(
#     output_dir="./models/llama3-finetuned",
#     num_train_epochs=2,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     logging_steps=10,
#     save_strategy="epoch",
#     learning_rate=2e-4,
#     fp16=True,
#     report_to=[],
# )


# # === Инициализация TRL-тренера
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     packing=False,
#     tokenizer=tokenizer,
#     peft_config=lora_config,
#     dataset_text_field="messages",  # TRL-SFT сам вызовет tokenizer.apply_chat_template
# )

# trainer.train()
# trainer.save_model("models/llama3-finetuned")

# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
# from peft import LoraConfig
# from trl import SFTTrainer
# import torch

# # === Модель LLaMA-3 ===
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # или OpenChat

# # === Датасет ===
# DATA_PATH = "data/llama_train.json"

# # === Загружаем jsonl
# dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# # === Сжимаем модель LoRA + 4bit
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, padding_side="right")
# tokenizer.pad_token = tokenizer.eos_token

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config,
#     device_map="auto"
# )

# training_args = TrainingArguments(
#     output_dir="./models/llama3-finetuned",
#     num_train_epochs=2,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     logging_steps=10,
#     save_strategy="epoch",
#     learning_rate=2e-4,
#     fp16=True,
#     report_to=[],
# )

# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     packing=False,
#     tokenizer=tokenizer,
#     peft_config=lora_config,
#     dataset_text_field="messages",  # TRL сам вызовет tokenizer.apply_chat_template
# )

# trainer.train()
# trainer.save_model("models/llama3-finetuned")

# print("✅ Обучение завершено. Модель сохранена в models/llama3-finetuned")

# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
# from peft import LoraConfig
# from trl import SFTTrainer
# import torch

# # ✅ Используем открытую модель
# MODEL_NAME = "openchat/openchat-3.5-0106"
# DATA_PATH = "data/llama_train.json"

# dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     torch_dtype=torch.float32
# )

# training_args = TrainingArguments(
#     output_dir="./models/openchat-finetuned",
#     num_train_epochs=2,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     logging_steps=10,
#     save_strategy="epoch",
#     learning_rate=2e-4,
#     fp16=True,
#     report_to=[],
# )

# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     dataset_text_field="messages",
#     peft_config=lora_config
# )

# trainer.train()
# trainer.save_model("models/openchat-finetuned")

# print("✅ Модель успешно дообучена и сохранена!")

# from trl import SFTTrainer
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# from peft import LoraConfig
# import torch

# MODEL_NAME = "openchat/openchat-3.5-0106"
# DATA_PATH = "data/llama_train.json"

# # Загрузка токенизатора и модели
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# )

# # Загрузка датасета
# dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# # Настройка PEFT
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# # Настройка обучения
# training_args = TrainingArguments(
#     output_dir="./models/llama3-finetuned",
#     num_train_epochs=2,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     logging_steps=10,
#     save_strategy="epoch",
#     learning_rate=2e-4,
#     fp16=True,
#     report_to=[],
# )

# # 👇 ВАЖНО: твоя функция, преобразующая messages[] в строку
# def formatting_func(example):
#     return tokenizer.apply_chat_template(example["messages"], tokenize=False)

# # ✅ Новый вызов SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     peft_config=lora_config,
#     formatting_func=formatting_func,
#     tokenizer=tokenizer,
# )

# # Обучение
# trainer.train()
# trainer.save_model("models/llama3-finetuned")

# print("✅ Обучение завершено.")

# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# from peft import LoraConfig
# from trl import SFTTrainer
# import torch

# MODEL_NAME = "openchat/openchat-3.5-0106"
# DATA_PATH = "data/llama_train.json"

# # Загружаем модель и токенизатор
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# )

# # Загружаем датасет
# dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# # Настройка LoRA
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# # Аргументы обучения
# training_args = TrainingArguments(
#     output_dir="./models/llama3-finetuned",
#     per_device_train_batch_size=1,
#     num_train_epochs=2,
#     gradient_accumulation_steps=4,
#     learning_rate=2e-4,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     report_to=[]
# )

# # 🔧 formatting_func для apply_chat_template (замена dataset_text_field)
# def format_func(example):
#     return tokenizer.apply_chat_template(example["messages"], tokenize=False)

# # ✅ Инициализация SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     formatting_func=format_func,
#     peft_config=lora_config
# )

# # ==== Обучаем модель ====
# trainer.train()

# # ==== Сохраняем ====
# trainer.save_model("models/llama3-finetuned")

# print("✅ Обучение завершено и модель сохранена.")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

MODEL_NAME = "openchat/openchat-3.5-0106"
DATA_PATH = "../data/llama_train.json"

# Загружаем модель и токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Загружаем модель на CPU/GPU, без квантования
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# 🔧 Добавляем LoRA-поддержку
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Загружаем датасет
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Аргументы обучения
training_args = TrainingArguments(
    output_dir="../models/llama3-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to=[]
)

# Функция для шаблонизации
def format_func(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

# Создаём тренер
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=format_func
)

trainer.train()
trainer.save_model("../models/llama3-finetuned")
tokenizer.save_pretrained("../models/llama3-complete")