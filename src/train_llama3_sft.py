from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import torch
import os
from transformers import Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.integrations import TensorBoardCallback

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH = "../data/llama_train.json"

def main():
    os.environ["WANDB_DISABLED"] = "true"  # отключаем wandb
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset("json", data_files="data/llama_train.json", split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    def format_func(example):
        return tokenizer.apply_chat_template(example["messages"], return_tensors="pt", truncation=True, max_length=1024)

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    training_args = TrainingArguments(
        output_dir="../models/llama3-finetuned",
        per_device_train_batch_size=1,
        num_train_epochs=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_dir="outputs/logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[]
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=format_func
    )

    print("🔧 Начинаем обучение...")
    trainer.train()

    print("✅ Сохраняем модель...")
    trainer.save_model("models/llama3-finetuned")

if __name__ == "__main__":
    main()