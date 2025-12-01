# AI Itinerary Generator - Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Fine-tuning Process](#fine-tuning-process)
4. [Data Management](#data-management)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [Adding New Data](#adding-new-data)
8. [Commands Reference](#commands-reference)
9. [Troubleshooting](#troubleshooting)

## Project Overview

### Architecture
- **Base Model**: `ozgecanaktas/tinyllama-itinerary-final`
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: Transformers, PEFT, PyTorch
- **Task**: Text Generation for Travel Itineraries

### Key Components
```
Local/
├── train.jsonl              # Training dataset
├── train_lora_mac.py       # LoRA fine-tuning script
├── infer_lora.py           # LoRA inference script
├── infer_generate_fixed.py # Basic inference script
├── infer_fixed.py          # Pipeline inference script
└── README.md               # Project documentation
```

## Model Architecture

### Base Model Details
- **Model Type**: TinyLlama (Causal Language Model)
- **Parameters**: ~1.1B parameters
- **Architecture**: Transformer decoder
- **Context Length**: 2048 tokens
- **Vocabulary Size**: 32,000 tokens

### LoRA Configuration
```python
# Typical LoRA settings for this project
lora_config = {
    "r": 16,                    # Rank of adaptation
    "lora_alpha": 32,          # LoRA scaling parameter
    "target_modules": ["q_proj", "v_proj"],  # Target attention layers
    "lora_dropout": 0.1,       # Dropout for LoRA layers
    "bias": "none",            # Bias type
    "task_type": "CAUSAL_LM"   # Task type
}
```

## Fine-tuning Process

### Prerequisites
```bash
# Install required dependencies
pip install transformers datasets peft accelerate torch bitsandbytes
```

### Training Data Format
The `train.jsonl` file uses instruction-following format:
```json
{
    "instruction": "Create a 1-day budget walking itinerary for Jaipur...",
    "input": "",
    "output": "8:00 AM - Breakfast: Kachori & chai near Johari Bazaar (₹30)..."
}
```

### Fine-tuning Steps

#### Step 1: Prepare Training Script
Create `train_lora_mac.py`:
```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Model configuration
MODEL_NAME = "ozgecanaktas/tinyllama-itinerary-final"
OUTPUT_DIR = "./lora-itinerary-model"

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=2,
    remove_unused_columns=False,
)
```

#### Step 2: Data Processing
```python
def format_instruction(example):
    """Format training examples for instruction following"""
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    
    return {
        "text": prompt + example["output"] + tokenizer.eos_token
    }

# Load and process dataset
dataset = load_dataset("json", data_files="train.jsonl")
dataset = dataset.map(format_instruction)
```

## Data Management

### Current Dataset Structure
The training data contains examples for:
- **Cities**: Jaipur, Noida
- **Budget Ranges**: ₹200 - ₹3000
- **Duration**: 1-5 days
- **Traveler Types**: Solo, family, couples, students
- **Themes**: Budget, cultural, food, photography, historical

### Data Quality Guidelines
1. **Real POIs Only**: Use actual places and attractions
2. **Accurate Costs**: Include realistic pricing with ~ for estimates
3. **Timeline Format**: Use consistent time formatting (HH:MM AM/PM)
4. **Budget Breakdown**: Provide clear cost calculations
5. **Transport Info**: Include public transport options

## Training Pipeline

### Complete Training Workflow

#### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Run Training
```bash
python3 train_lora_mac.py
```

#### 3. Monitor Training
```bash
# Check training logs
tail -f ./lora-itinerary-model/trainer_state.json

# Monitor GPU usage (if using GPU)
nvidia-smi -l 1
```

### Training Parameters Explained
- **Learning Rate**: 2e-4 (optimal for LoRA fine-tuning)
- **Batch Size**: 1 (memory efficient for local training)
- **Gradient Accumulation**: 4 (effective batch size = 4)
- **Epochs**: 3 (prevents overfitting on small dataset)
- **LoRA Rank**: 16 (balance between performance and efficiency)

## Inference Pipeline

### Method 1: LoRA Inference (Recommended)
```python
# infer_lora.py
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model and LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("ozgecanaktas/tinyllama-itinerary-final")
model = PeftModel.from_pretrained(base_model, "./lora-itinerary-model")
tokenizer = AutoTokenizer.from_pretrained("ozgecanaktas/tinyllama-itinerary-final")

# Generate itinerary
prompt = "### Instruction:\nCreate a 2-day budget itinerary for Delhi under ₹1000\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=300)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Method 2: Pipeline Inference
```python
# infer_fixed.py - Uses pipeline for simpler inference
from transformers import pipeline

pipe = pipeline("text-generation", 
               model="ozgecanaktas/tinyllama-itinerary-final",
               tokenizer="ozgecanaktas/tinyllama-itinerary-final")

result = pipe(prompt, max_new_tokens=250, do_sample=False)
```

## Adding New Data

### Step-by-Step Process

#### 1. Prepare New Training Examples
Add entries to `train.jsonl` following this format:
```json
{"instruction":"Create a 2-day budget itinerary for Mumbai under ₹1200","input":"","output":"Day 1:\n- Morning: Gateway of India (free)\n- Afternoon: Colaba Causeway shopping (₹200)\n- Evening: Marine Drive walk (free)\nDay 2:\n- Morning: Elephanta Caves (ferry ₹150, entry ₹40)\n- Afternoon: Crawford Market (₹100)\nTotal: ₹490"}
```

#### 2. Validate Data Quality
```bash
# Check JSON format
python3 -c "
import json
with open('train.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Error in line {i+1}: {e}')
"
```

#### 3. Retrain Model
```bash
# Backup previous model
mv ./lora-itinerary-model ./lora-itinerary-model-backup-$(date +%Y%m%d)

# Run training with new data
python3 train_lora_mac.py
```

#### 4. Test New Model
```bash
python3 infer_lora.py
```

### Data Expansion Guidelines

#### Adding New Cities
1. Research real POIs and attractions
2. Get accurate pricing information
3. Include local transport options
4. Add 5-10 examples per city
5. Cover different budget ranges

#### Adding New Themes
- Adventure travel
- Luxury budget (₹5000+)
- Accessibility-focused
- Monsoon/seasonal itineraries
- Business travel

## Commands Reference

### Training Commands
```bash
# Basic training
python3 train_lora_mac.py

# Training with custom parameters
python3 train_lora_mac.py --epochs 5 --learning_rate 1e-4

# Resume training from checkpoint
python3 train_lora_mac.py --resume_from_checkpoint ./lora-itinerary-model/checkpoint-100
```

### Inference Commands
```bash
# LoRA inference
python3 infer_lora.py

# Basic inference
python3 infer_generate_fixed.py

# Pipeline inference
python3 infer_fixed.py
```

### Data Management Commands
```bash
# Validate training data
python3 -m json.tool train.jsonl > /dev/null && echo "Valid JSON"

# Count training examples
wc -l train.jsonl

# Backup training data
cp train.jsonl train_backup_$(date +%Y%m%d).jsonl

# Split data for validation (optional)
head -n 80 train.jsonl > train_split.jsonl
tail -n +81 train.jsonl > val_split.jsonl
```

### Model Management Commands
```bash
# Check model size
du -sh ./lora-itinerary-model/

# List model files
ls -la ./lora-itinerary-model/

# Merge LoRA with base model (for deployment)
python3 merge_lora.py
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Solution: Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
# Or modify training_args: per_device_train_batch_size=1
```

#### 2. JSON Format Errors
```bash
# Check for malformed JSON
python3 -c "
import json
with open('train.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line.strip())
        except:
            print(f'Error at line {i}: {line}')
"
```

#### 3. Model Loading Issues
```python
# Add error handling
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Model loading failed: {e}")
    # Fallback to CPU or different model
```

#### 4. Poor Generation Quality
- Increase training epochs (3-5)
- Add more diverse training examples
- Adjust LoRA rank (8-32)
- Modify generation parameters (temperature, top_p)

### Performance Optimization

#### Memory Optimization
```python
# Use gradient checkpointing
training_args.gradient_checkpointing = True

# Use 8-bit training
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)
```

#### Speed Optimization
```python
# Use flash attention (if available)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="flash_attention_2"
)
```

## Deployment Considerations

### Model Export
```python
# Merge LoRA adapter with base model
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("ozgecanaktas/tinyllama-itinerary-final")
model = PeftModel.from_pretrained(base_model, "./lora-itinerary-model")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-itinerary-model")
```

### API Integration
```python
# Simple Flask API example
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
pipe = pipeline("text-generation", model="./merged-itinerary-model")

@app.route('/generate', methods=['POST'])
def generate_itinerary():
    prompt = request.json['prompt']
    result = pipe(prompt, max_new_tokens=300)
    return jsonify({'itinerary': result[0]['generated_text']})
```

## Future Enhancements

### Planned Features
1. **Multi-language Support**: Hindi, regional languages
2. **Real-time Pricing**: API integration for current prices
3. **Weather Integration**: Season-appropriate suggestions
4. **User Preferences**: Learning from user feedback
5. **Image Generation**: Visual itinerary cards

### Technical Improvements
1. **Quantization**: 4-bit/8-bit model compression
2. **Streaming**: Real-time generation
3. **Caching**: Response caching for common queries
4. **Monitoring**: Performance and usage analytics

---

*Last Updated: $(date)*
*Version: 1.0*