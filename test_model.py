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

def test_itinerary(prompt_text):
    prompt = f"### Instruction:\n{prompt_text}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result.startswith(prompt):
        result = result[len(prompt):].strip()
    return result

# Test different cities and scenarios
test_cases = [
    "Create a 1-day budget itinerary for Delhi under ₹300 with street food and historical sites.",
    "Generate a 2-day Jaipur itinerary for ₹800 focusing on photography spots.",
    "Create a 1-day Noida itinerary for ₹200 with shopping and modern attractions.",
    "Design a 3-day Delhi food trail covering famous markets, budget ₹1000."
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n=== TEST {i}: {test_case[:50]}... ===")
    result = test_itinerary(test_case)
    print(result)
    print("-" * 80)