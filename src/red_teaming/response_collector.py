import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import json
from datetime import datetime
from .prompt_generator import AdversarialPrompt

class LLMResponseCollector:
    """
    Collects responses from various open-source LLMs for red teaming analysis.
    """
    def __init__(self, model_name: str = "mistralai/Devstral-Small-2505"):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, falling back to microsoft/DialoGPT-medium")
            fallback_model = "microsoft/DialoGPT-medium"
            self.model_name = fallback_model
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate a response to the given prompt."""
        try:
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def collect_responses(self, prompts: List[AdversarialPrompt], num_responses_per_prompt: int = 1) -> List[Dict]:
        """Collect responses for a list of adversarial prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            
            for response_num in range(num_responses_per_prompt):
                response = self.generate_response(prompt.text)
                
                result = {
                    "prompt_id": f"{i}_{response_num}",
                    "prompt_text": prompt.text,
                    "prompt_category": prompt.category,
                    "prompt_subcategory": prompt.subcategory,
                    "expected_harm_level": prompt.expected_harm_level,
                    "techniques_used": prompt.techniques_used,
                    "response_text": response,
                    "model_name": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "generation_params": {
                        "temperature": 0.7,
                        "max_length": 200
                    }
                }
                
                results.append(result)
        
        return results
    
    def save_responses(self, responses: List[Dict], filename: str = "generated_responses.json"):
        """Save collected responses to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(responses)} responses to {filename}")