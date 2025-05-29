import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.safety_filter_classifier.classifier import SafetyClassifier
import random
from collections import defaultdict

class RejectionSampler:
    """
    Implements rejection sampling for LLM safety.
    Generates multiple responses and selects the safest one based on a safety classifier.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 safety_classifier_path: Optional[str] = None,
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load generation model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load safety classifier
        if safety_classifier_path:
            self.safety_classifier = SafetyClassifier.load_model(safety_classifier_path)
            self.safety_classifier.eval()
        else:
            print("Warning: No safety classifier provided. Using rule-based fallback.")
            self.safety_classifier = None
        
        # Rejection criteria
        self.safety_threshold = 0.7  # Minimum confidence for safe classification
        self.max_attempts = 10  # Maximum generation attempts
        
        # Generation parameters for diversity
        self.generation_configs = [
            {"temperature": 0.7, "top_p": 0.9, "do_sample": True},
            {"temperature": 0.5, "top_p": 0.8, "do_sample": True},
            {"temperature": 0.9, "top_p": 0.95, "do_sample": True},
            {"temperature": 0.3, "top_p": 0.7, "do_sample": True},
            {"temperature": 1.0, "top_p": 0.9, "do_sample": True}
        ]
    
    def generate_single_response(self, prompt: str, config: Dict, max_length: int = 200) -> str:
        """Generate a single response with given configuration."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    **config
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Generation error: {str(e)}"
    
    def evaluate_safety(self, prompt: str, response: str) -> Tuple[str, float, Dict]:
        """Evaluate the safety of a response."""
        if self.safety_classifier:
            # Use trained classifier
            combined_text = f"Prompt: {prompt} Response: {response}"
            label, confidence = self.safety_classifier.predict(combined_text)
            
            safety_score = confidence if label == 'safe' else (1 - confidence)
            
            return label, safety_score, {
                'classifier_label': label,
                'classifier_confidence': confidence,
                'safety_score': safety_score
            }
        else:
            # Fallback to rule-based evaluation
            return self._rule_based_safety_check(prompt, response)
    
    def _rule_based_safety_check(self, prompt: str, response: str) -> Tuple[str, float, Dict]:
        """Fallback rule-based safety evaluation."""
        response_lower = response.lower()
        
        # Check for explicit refusal (very safe)
        refusal_patterns = [
            "i can't", "i cannot", "i'm not able", "i won't", "i shouldn't",
            "not appropriate", "against guidelines", "not comfortable"
        ]
        
        if any(pattern in response_lower for pattern in refusal_patterns):
            return 'safe', 0.95, {'rule': 'refusal_detected'}
        
        # Check for harmful content
        harmful_keywords = [
            "violence", "harm", "hurt", "kill", "destroy", "attack",
            "illegal", "crime", "fraud", "hate", "abuse"
        ]
        
        harm_count = sum(1 for keyword in harmful_keywords if keyword in response_lower)
        if harm_count >= 2:
            return 'unsafe_abusive', 0.8, {'rule': f'harmful_keywords_count_{harm_count}'}
        
        # Check for bias indicators
        bias_patterns = [
            r"\ball\s+(women|men|people)\s+are",
            r"\b(naturally|inherently)\s+(better|worse)",
            r"\b(can't|cannot)\s+understand"
        ]
        
        import re
        bias_matches = sum(1 for pattern in bias_patterns if re.search(pattern, response_lower))
        if bias_matches >= 1:
            return 'biased_harmful', 0.7, {'rule': f'bias_patterns_{bias_matches}'}
        
        # Check response quality (very short or error responses)
        if len(response.strip()) < 10 or "error" in response_lower:
            return 'safe', 0.6, {'rule': 'short_or_error_response'}
        
        # Default to moderately safe
        return 'safe', 0.5, {'rule': 'no_clear_issues'}
    
    def rejection_sample(self, prompt: str, 
                        num_candidates: int = 5, 
                        max_attempts: int = None) -> Dict:
        """
        Generate multiple responses and select the safest one.
        
        Returns:
            Dict containing the best response and sampling statistics
        """
        max_attempts = max_attempts or self.max_attempts
        candidates = []
        generation_stats = defaultdict(int)
        
        attempt = 0
        while len(candidates) < num_candidates and attempt < max_attempts:
            # Select random generation configuration
            config = random.choice(self.generation_configs)
            
            # Generate response
            response = self.generate_single_response(prompt, config)
            generation_stats['total_attempts'] += 1
            
            # Evaluate safety
            label, safety_score, eval_details = self.evaluate_safety(prompt, response)
            
            candidate = {
                'response': response,
                'safety_label': label,
                'safety_score': safety_score,
                'eval_details': eval_details,
                'generation_config': config,
                'attempt': attempt
            }
            
            candidates.append(candidate)
            generation_stats[f'label_{label}'] += 1
            
            # Early termination if we find a highly safe response
            if label == 'safe' and safety_score > 0.9:
                generation_stats['early_termination'] = True
                break
            
            attempt += 1
        
        # Select best candidate
        if not candidates:
            return {
                'selected_response': "I apologize, but I cannot generate a safe response to this prompt.",
                'safety_label': 'safe',
                'safety_score': 1.0,
                'candidates': [],
                'selection_method': 'fallback',
                'generation_stats': dict(generation_stats)
            }
        
        # Sort by safety score (descending) and select best
        candidates.sort(key=lambda x: x['safety_score'], reverse=True)
        best_candidate = candidates[0]
        
        # Additional selection criteria
        selection_method = 'safety_score'
        
        # If multiple candidates have similar safety scores, prefer longer, more informative responses
        if len(candidates) > 1:
            top_candidates = [c for c in candidates if c['safety_score'] >= best_candidate['safety_score'] - 0.1]
            if len(top_candidates) > 1:
                # Among similarly safe responses, prefer longer ones (but not too long)
                scored_candidates = []
                for c in top_candidates:
                    length_score = min(len(c['response']) / 200.0, 1.0)  # Normalize to [0,1], cap at 200 chars
                    combined_score = c['safety_score'] * 0.8 + length_score * 0.2
                    scored_candidates.append((c, combined_score))
                
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidate = scored_candidates[0][0]
                selection_method = 'combined_safety_length'
        
        return {
            'selected_response': best_candidate['response'],
            'safety_label': best_candidate['safety_label'],
            'safety_score': best_candidate['safety_score'],
            'eval_details': best_candidate['eval_details'],
            'candidates': candidates,
            'selection_method': selection_method,
            'generation_stats': dict(generation_stats)
        }
    
    def batch_rejection_sample(self, prompts: List[str], 
                              num_candidates: int = 5) -> List[Dict]:
        """Apply rejection sampling to a batch of prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.rejection_sample(prompt, num_candidates)
            result['prompt'] = prompt
            result['prompt_id'] = i
            results.append(result)
        
        return results
    
    def analyze_sampling_effectiveness(self, results: List[Dict]) -> Dict:
        """Analyze the effectiveness of rejection sampling."""
        if not results:
            return {}
        
        analysis = {
            'total_prompts': len(results),
            'safety_improvement': {},
            'generation_efficiency': {},
            'label_distribution': defaultdict(int)
        }
        
        # Calculate safety improvements
        original_safe_count = 0
        final_safe_count = 0
        
        for result in results:
            candidates = result.get('candidates', [])
            if candidates:
                # Compare first vs selected response
                first_candidate = candidates[0] if candidates else None
                selected_safety = result.get('safety_score', 0)
                first_safety = first_candidate.get('safety_score', 0) if first_candidate else 0
                
                if first_candidate and first_candidate['safety_label'] == 'safe':
                    original_safe_count += 1
                
                if result.get('safety_label') == 'safe':
                    final_safe_count += 1
                
                # Track label distribution
                analysis['label_distribution'][result.get('safety_label', 'unknown')] += 1
        
        analysis['safety_improvement'] = {
            'original_safe_responses': original_safe_count,
            'final_safe_responses': final_safe_count,
            'improvement_count': final_safe_count - original_safe_count,
            'improvement_rate': (final_safe_count - original_safe_count) / len(results) if results else 0
        }
        
        # Generation efficiency
        total_attempts = sum(r['generation_stats'].get('total_attempts', 0) for r in results)
        analysis['generation_efficiency'] = {
            'average_attempts_per_prompt': total_attempts / len(results) if results else 0,
            'total_generations': total_attempts,
            'early_terminations': sum(1 for r in results if r['generation_stats'].get('early_termination', False))
        }
        
        return analysis

class AdaptiveRejectionSampler(RejectionSampler):
    """
    Enhanced rejection sampler that adapts generation parameters based on prompt risk.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Risk-based generation strategies
        self.risk_strategies = {
            'low_risk': {
                'num_candidates': 3,
                'generation_configs': [
                    {"temperature": 0.7, "top_p": 0.9, "do_sample": True},
                    {"temperature": 0.5, "top_p": 0.8, "do_sample": True}
                ]
            },
            'medium_risk': {
                'num_candidates': 5,
                'generation_configs': [
                    {"temperature": 0.3, "top_p": 0.7, "do_sample": True},
                    {"temperature": 0.5, "top_p": 0.8, "do_sample": True},
                    {"temperature": 0.7, "top_p": 0.9, "do_sample": True}
                ]
            },
            'high_risk': {
                'num_candidates': 8,
                'generation_configs': [
                    {"temperature": 0.1, "top_p": 0.5, "do_sample": True},
                    {"temperature": 0.3, "top_p": 0.6, "do_sample": True},
                    {"temperature": 0.5, "top_p": 0.7, "do_sample": True}
                ]
            }
        }
    
    def assess_prompt_risk(self, prompt: str) -> str:
        """Assess the risk level of a prompt."""
        prompt_lower = prompt.lower()
        
        # High-risk indicators
        high_risk_keywords = [
            'how to', 'teach me', 'show me', 'explain how',
            'violence', 'harm', 'illegal', 'hack', 'bypass',
            'ignore', 'pretend', 'roleplay', 'jailbreak'
        ]
        
        # Medium-risk indicators
        medium_risk_keywords = [
            'opinion', 'think about', 'what about', 'typical',
            'stereotype', 'differences between', 'better than'
        ]
        
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in prompt_lower)
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in prompt_lower)
        
        if high_risk_count >= 2:
            return 'high_risk'
        elif high_risk_count >= 1 or medium_risk_count >= 2:
            return 'medium_risk'
        else:
            return 'low_risk'
    
    def adaptive_rejection_sample(self, prompt: str) -> Dict:
        """Apply adaptive rejection sampling based on prompt risk."""
        risk_level = self.assess_prompt_risk(prompt)
        strategy = self.risk_strategies[risk_level]
        
        # Temporarily override generation configs
        original_configs = self.generation_configs
        self.generation_configs = strategy['generation_configs']
        
        # Perform rejection sampling
        result = self.rejection_sample(prompt, strategy['num_candidates'])
        result['risk_level'] = risk_level
        result['strategy_used'] = strategy
        
        # Restore original configs
        self.generation_configs = original_configs
        
        return result