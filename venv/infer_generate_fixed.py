# # infer_generate_fixed.py
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# MODEL = "microsoft/DialoGPT-small"

# print("Loading tokenizer and model (CPU)...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
# model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=False, device_map={"": "cpu"}, torch_dtype=torch.float32)

# prompt = (
#     "### Instruction:\n"
#     "Create a 1-day itinerary for Noida city (NOT Jaipur) for one person with a total budget of ₹200. "
#     "Include only Noida attractions like DLF Mall of India, Worlds of Wonder, Okhla Bird Sanctuary, Sector 18 market, etc. "
#     "Do NOT include any Jaipur places like Amer Fort, Hawa Mahal, etc. "
#     "Provide a timeline and approximate costs for Noida only. Output only the itinerary.\n\n### Response:\n"
# )

# # Tokenize prompt
# inputs = tokenizer(prompt, return_tensors="pt")
# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]

# # Generation params — explicit
# gen = model.generate(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     max_new_tokens=300,
#     do_sample=False,
#     temperature=0.0,
#     top_p=0.95,
#     eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
#     pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
# )

# out = tokenizer.decode(gen[0], skip_special_tokens=True)
# # remove the prompt part if model echoes it
# if out.startswith(prompt):
#     out = out[len(prompt):].strip()
# print("\n--- GENERATED ---\n")
# print(out)
