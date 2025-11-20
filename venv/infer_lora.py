# infer_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "ozgecanaktas/tinyllama-itinerary-final"
LORA_ADAPTER = "lora_tinyllama_itinerary"

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=False, device_map={"": "cpu"}, torch_dtype=torch.float32)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)

prompt = (
    "### Instruction:\n"
    "Create a 5-day Jaipur itinerary for one person with a total budget of ₹5000. "
    "Use only real Jaipur attractions and places. "
    "Provide a timeline and approximate costs. Output only the itinerary (do not repeat the prompt).\n\n### Response:\n"
)

# Tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Generation
gen = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
)

out = tokenizer.decode(gen[0], skip_special_tokens=True)
if out.startswith(prompt):
    out = out[len(prompt):].strip()
print("\n--- GENERATED OUTPUT ---\n")
print(out)