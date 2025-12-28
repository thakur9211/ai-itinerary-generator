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
    "Create a 2 days itinerary for Noida for one person with a total budget of ₹500. Use public transport where possible and include per-day cost breakdown.\n\n"
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

print("\n--- GENERATED NOIDA ITINERARY ---\n")
print(result)