from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "ozgecanaktas/tinyllama-itinerary-final"
LORA_MODEL_PATH = "./lora-itinerary-model"

print("Loading base model and LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=False,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)

# Load LoRA model
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

# Test with Delhi itinerary
prompt = (
    "### Instruction:\n"
    "Create a 2-day budget itinerary for Delhi under ₹1000 focusing on historical sites and street food. "
    "Use only real Delhi POIs like Red Fort, Jama Masjid, Humayun's Tomb, Qutub Minar, India Gate. "
    "Provide timeline and costs.\n\n"
    "### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs, 
    max_new_tokens=300,
    do_sample=False,
    temperature=0.0,
    pad_token_id=tokenizer.eos_token_id,
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove prompt if echoed
if result.startswith(prompt):
    result = result[len(prompt):].strip()

print("\n--- GENERATED DELHI ITINERARY ---\n")
print(result)