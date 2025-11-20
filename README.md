# AI Itinerary Generator

A fine-tuned AI model that generates travel itineraries for cities based on budget and preferences.

## Features

- Generate budget-friendly itineraries
- Customizable budget constraints
- Real attractions and places
- Timeline and cost breakdown
- Fine-tuned using LoRA (Low-Rank Adaptation)

## Files

- `train.jsonl` - Training data with Jaipur and Noida itinerary examples
- `train_lora_mac.py` - Fine-tuning script using LoRA
- `infer_lora.py` - Inference script for the fine-tuned model
- `infer_generate_fixed.py` - Basic inference script

## Setup

1. Install dependencies:
```bash
pip install transformers datasets peft accelerate torch
```

2. Fine-tune the model:
```bash
python3 train_lora_mac.py
```

3. Generate itineraries:
```bash
python3 infer_lora.py
```

## Usage

Edit the prompt in `infer_lora.py` to specify:
- City (Jaipur or Noida)
- Budget amount
- Duration (1-day, 3-day, etc.)
- Special requirements

## Model

Based on `ozgecanaktas/tinyllama-itinerary-final` with LoRA fine-tuning for multi-city support.