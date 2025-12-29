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
        
        try:
            self.model = PeftModel.from_pretrained(base_model, self.LORA_MODEL_PATH)
            print("LoRA model loaded successfully!")
        except Exception as e:
            print(f"LoRA loading failed: {e}. Using base model.")
            self.model = base_model
        
        print("Model loaded successfully!")
    
    def generate_itinerary(self, days, city, traveler_type, budget):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Model tokenizing input...")
        
        prompt = f"Create a {days} budget itinerary for {city} for {traveler_type} with ₹{budget} total budget:\n\n"
        
        print(f"[{timestamp}] Prompt: '{prompt}'")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        gen_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{gen_timestamp}] Model generating response...")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=200,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        decode_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{decode_timestamp}] Model decoding output...")
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[{decode_timestamp}] Raw model output: '{result}'")
        
        # Remove prompt if echoed and handle empty generation
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
            print(f"[{decode_timestamp}] After removing prompt: '{result}'")
        
        # If still empty, return a fallback message
        if not result or len(result.strip()) == 0:
            result = f"Day 1: Visit local attractions in {city}. Budget: ₹{budget}. Use public transport for cost-effective travel."
            print(f"[{decode_timestamp}] Using fallback response")
        
        print(f"[{decode_timestamp}] Final result: '{result}'")
        
        return result