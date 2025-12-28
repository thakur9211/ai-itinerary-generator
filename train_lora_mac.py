import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import json

# Model configuration
MODEL_NAME = "ozgecanaktas/tinyllama-itinerary-final"
OUTPUT_DIR = "./lora-itinerary-model"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=False,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)

# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def format_instruction(example):
    """Format training examples for instruction following"""
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    
    text = prompt + example["output"] + tokenizer.eos_token
    return {"text": text}

def tokenize_function(examples):
    """Tokenize the text"""
    result = tokenizer(examples["text"], truncation=True, padding=False, max_length=512)
    result["labels"] = result["input_ids"].copy()
    return result

# Load and process dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="rules_based_train.jsonl")
dataset = dataset.map(format_instruction)
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "input", "output", "text"])

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=False,  # Set to False for CPU training
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    remove_unused_columns=False,
    dataloader_drop_last=False,
    warmup_steps=10,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training completed!")