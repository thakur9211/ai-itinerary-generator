from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL = "ozgecanaktas/tinyllama-itinerary-final"

print("torch.cuda.is_available:", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=False,
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = (
    "Create a 1-day Jaipur itinerary under ₹200 using only real POIs such as "
    "Amer Fort, Hawa Mahal, Jantar Mantar, City Palace, Nahargarh Fort, Jal Mahal, "
    "Albert Hall, Birla Mandir, Bapu Bazaar, Galta Ji. "
    "Include timeline & costs."
)

out = pipe(prompt, max_new_tokens=250, do_sample=False)

print("\n--- OUTPUT ---\n")
print(out[0]["generated_text"])

