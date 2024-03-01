import datasets
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer


def formatting_func(example):
    text = f"<start_of_turn>user\n{example['INSTRUCTION'][0]}<end_of_turn> <start_of_turn>model\n{example['RESPONSE'][0]}<end_of_turn>"
    return [text]


model_id = "google/gemma-2b-it"

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

data = datasets.load_dataset("ahmed000000000/cybersec")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
model = AutoModelForCausalLM.from_pretrained(model_id, token="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")

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=150,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="venv/outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)

trainer.train()

text = """<start_of_turn>user
Explain 'AMSI init bypass' and its purpose.<end_of_turn>
<start_of_turn>model"""
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
