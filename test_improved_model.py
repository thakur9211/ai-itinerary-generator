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

def generate_itinerary(city, days, traveler_type, budget):
    """Generate itinerary with proper formatting"""
    
    prompt = (
        f"### Instruction:\n"
        f"Create a {days} itinerary for {city} for {traveler_type} with a total budget of ₹{budget}. "
        f"Use public/local transport and include per-day cost breakdown where applicable.\n\n"
        f"### Response:\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=500,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt if echoed
    if result.startswith(prompt):
        result = result[len(prompt):].strip()
    
    return result

# Test cases
test_cases = [
    ("Noida", "2 days", "solo", "500"),
    ("Delhi", "3 days", "couple (2)", "2000"),
    ("Mumbai", "1 day", "family (3-4)", "1200"),
    ("Bangalore", "2 days", "solo", "800")
]

for city, days, traveler_type, budget in test_cases:
    print(f"\n{'='*60}")
    print(f"TESTING: {days} in {city} for {traveler_type} with ₹{budget}")
    print(f"{'='*60}")
    
    result = generate_itinerary(city, days, traveler_type, budget)
    print(result)
    print(f"{'='*60}")