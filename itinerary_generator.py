from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

class ItineraryGenerator:
    def __init__(self):
        self.MODEL_NAME = "ozgecanaktas/tinyllama-itinerary-final"
        self.LORA_MODEL_PATH = "./lora-itinerary-model"
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        print("Loading base model and LoRA adapter...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=False, use_fast=False, legacy=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=False,
            device_map={"": "cpu"},
            torch_dtype=torch.float32
        )
        self.model = PeftModel.from_pretrained(base_model, self.LORA_MODEL_PATH)
        print("Model loaded successfully!")
    
    def generate_itinerary(self, days, city, traveler_type, budget):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Model tokenizing input...")
        
        prompt = (
            f"Create a {days} itinerary for {city} for {traveler_type} with a total budget of ₹{budget}. "
            f"Use public transport and include per-day cost breakdown."
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        gen_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{gen_timestamp}] Model generating response...")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        decode_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{decode_timestamp}] Model decoding output...")
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt if echoed
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
        
        return result