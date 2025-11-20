# infer_generate_fixed.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "ozgecanaktas/tinyllama-itinerary-final"

print("Loading tokenizer and model (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=False,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)

prompt = (
    "### Instruction:\n"
    "Create a 1-day Jaipur itinerary for one person with a total budget of ₹200. "
    "Use only real Jaipur POIs: Amer Fort, Hawa Mahal, Jantar Mantar, City Palace, "
    "Nahargarh Fort, Jal Mahal, Albert Hall, Birla Mandir, Bapu Bazaar, Galta Ji. "
    "Provide a timeline and approximate costs. Output only the itinerary.\n\n"
    "### Response:\n"
)

# Tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt")

gen = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=False,
    temperature=0.0,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

text = tokenizer.decode(gen[0], skip_special_tokens=True)

# Remove prompt if echoed
if text.startswith(prompt):
    text = text[len(prompt):].strip()

print("\n--- GENERATED OUTPUT ---\n")
print(text)

