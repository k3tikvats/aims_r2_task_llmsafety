import re
import json
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dataclasses import dataclass
from enum import Enum

class PromptRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PromptTransformation:
    original_prompt: str
    transformed_prompt: str
    transformation_type: str
    risk_reduction: float
    confidence: float
    explanation: str

class PromptUpdater:
    """
    Transforms adversarial or risky prompts into safer versions while preserving legitimate intent.
    Implements multiple transformation strategies to neutralize harmful prompts.
    """
    
    def __init__(self, 
                 paraphrasing_model: str = "t5-base",
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load paraphrasing model for sophisticated transformations
        try:
            self.paraphraser_tokenizer = AutoTokenizer.from_pretrained(paraphrasing_model)
            self.paraphraser_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrasing_model)
            self.paraphraser_model.to(self.device)
        except:
            print("Warning: Could not load paraphrasing model. Using rule-based transformations only.")
            self.paraphraser_tokenizer = None
            self.paraphraser_model = None
        
        # Risk indicators and their transformation strategies
        self.risk_patterns = {
            'jailbreak_attempts': {
                'patterns': [
                    r'ignore\s+(?:previous\s+)?instructions',
                    r'pretend\s+you\s+are\s+(?:not\s+)?(?:an?\s+)?ai',
                    r'roleplay\s+as',
                    r'act\s+like\s+you\s+are',
                    r'you\s+are\s+now\s+dan',
                    r'forget\s+your\s+guidelines',
                    r'bypass\s+your\s+(?:safety\s+)?(?:rules|guidelines)',
                    r'developer\s+mode',
                    r'unrestricted\s+mode'
                ],
                'transformations': ['remove_jailbreak', 'neutralize_roleplay']
            },
            'manipulation_tactics': {
                'patterns': [
                    r'for\s+(?:my\s+)?(?:research|homework|story|novel)',
                    r'hypothetically',
                    r'in\s+a\s+fictional\s+world',
                    r'just\s+curious',
                    r'asking\s+for\s+a\s+friend',
                    r'what\s+if\s+someone'
                ],
                'transformations': ['clarify_intent', 'add_safety_context']
            },
            'harmful_instructions': {
                'patterns': [
                    r'how\s+to\s+(?:harm|hurt|kill|destroy)',
                    r'teach\s+me\s+to\s+(?:hack|steal|fraud)',
                    r'show\s+me\s+how\s+to\s+(?:break|bypass)',
                    r'guide\s+to\s+(?:illegal|harmful)',
                    r'step\s+by\s+step\s+(?:violence|crime)'
                ],
                'transformations': ['redirect_to_safety', 'educational_reframe']
            },
            'bias_elicitation': {
                'patterns': [
                    r'all\s+(?:women|men|people)\s+are',
                    r'why\s+are\s+(?:women|men)\s+(?:better|worse)',
                    r'typical\s+(?:female|male)\s+behavior',
                    r'naturally\s+(?:women|men)\s+are',
                    r'prove\s+that\s+(?:group)\s+is'
                ],
                'transformations': ['neutralize_bias', 'promote_equality']
            }
        }
        
        # Safe transformation templates
        self.transformation_templates = {
            'neutralize_request': [
                "Can you provide factual information about {}?",
                "I'm interested in learning about {} in an educational context.",
                "Could you explain the topic of {} objectively?",
                "What are the key facts about {}?"
            ],
            'safety_focused': [
                "What are the safety considerations regarding {}?",
                "How can one approach {} responsibly and ethically?",
                "What are the legal and ethical aspects of {}?",
                "Can you discuss {} from a harm reduction perspective?"
            ],
            'educational_reframe': [
                "I'm researching {} for educational purposes. What should I know?",
                "Could you provide an academic overview of {}?",
                "What would be important to understand about {} from a scholarly perspective?",
                "Can you explain {} in an informative and balanced way?"
            ]
        }
    
    def assess_prompt_risk(self, prompt: str) -> Tuple[PromptRiskLevel, Dict[str, List[str]]]:
        """Assess the risk level of a prompt and identify concerning patterns."""
        detected_patterns = {}
        total_risk_score = 0
        
        prompt_lower = prompt.lower()
        
        for risk_type, config in self.risk_patterns.items():
            matches = []
            for pattern in config['patterns']:
                if re.search(pattern, prompt_lower):
                    matches.append(pattern)
            
            if matches:
                detected_patterns[risk_type] = matches
                # Weight different risk types
                risk_weights = {
                    'jailbreak_attempts': 3,
                    'harmful_instructions': 4,
                    'manipulation_tactics': 2,
                    'bias_elicitation': 2
                }
                total_risk_score += len(matches) * risk_weights.get(risk_type, 1)
        
        # Determine risk level
        if total_risk_score >= 8:
            risk_level = PromptRiskLevel.CRITICAL
        elif total_risk_score >= 5:
            risk_level = PromptRiskLevel.HIGH
        elif total_risk_score >= 2:
            risk_level = PromptRiskLevel.MEDIUM
        else:
            risk_level = PromptRiskLevel.LOW
        
        return risk_level, detected_patterns
    
    def extract_legitimate_intent(self, prompt: str) -> str:
        """Extract the legitimate underlying intent from a potentially problematic prompt."""
        # Remove obvious jailbreak attempts
        cleaned = re.sub(r'ignore\s+(?:previous\s+)?instructions[^.]*\.?\s*', '', prompt, flags=re.IGNORECASE)
        cleaned = re.sub(r'pretend\s+you\s+are[^.]*\.?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'roleplay\s+as[^.]*\.?\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove manipulation framing
        cleaned = re.sub(r'(?:for\s+my\s+)?(?:research|homework|story)[^.]*\.?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'hypothetically[^,]*,?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'asking\s+for\s+a\s+friend[^.]*\.?\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Extract core question/topic
        # Look for question patterns
        question_patterns = [
            r'(?:how|what|why|when|where|who)\s+[^?]*\?',
            r'(?:can|could|would|should|will)\s+you\s+[^?]*\?',
            r'(?:explain|describe|tell\s+me\s+about)\s+[^.?]*[.?]?'
        ]
        
        for pattern in question_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return match.group().strip()
        
        # Fallback: return cleaned prompt
        return cleaned.strip()
    
    def apply_rule_based_transformations(self, prompt: str, detected_patterns: Dict[str, List[str]]) -> List[PromptTransformation]:
        """Apply rule-based transformations to mitigate risks."""
        transformations = []
        
        for risk_type, patterns in detected_patterns.items():
            config = self.risk_patterns[risk_type]
            
            for transformation_type in config['transformations']:
                transformed = self._apply_specific_transformation(prompt, risk_type, transformation_type)
                
                if transformed != prompt:
                    risk_reduction = self._estimate_risk_reduction(prompt, transformed)
                    
                    transformation = PromptTransformation(
                        original_prompt=prompt,
                        transformed_prompt=transformed,
                        transformation_type=transformation_type,
                        risk_reduction=risk_reduction,
                        confidence=0.8,
                        explanation=f"Applied {transformation_type} to address {risk_type}"
                    )
                    transformations.append(transformation)
        
        return transformations
    
    def _apply_specific_transformation(self, prompt: str, risk_type: str, transformation_type: str) -> str:
        """Apply a specific transformation based on risk type and strategy."""
        
        if transformation_type == 'remove_jailbreak':
            # Remove jailbreak instructions
            transformed = prompt
            jailbreak_patterns = [
                r'ignore\s+(?:previous\s+)?instructions[^.]*\.?\s*',
                r'pretend\s+you\s+are[^.]*\.?\s*',
                r'roleplay\s+as[^.]*\.?\s*',
                r'act\s+like\s+you\s+are[^.]*\.?\s*',
                r'you\s+are\s+now\s+dan[^.]*\.?\s*',
                r'forget\s+your\s+guidelines[^.]*\.?\s*'
            ]
            
            for pattern in jailbreak_patterns:
                transformed = re.sub(pattern, '', transformed, flags=re.IGNORECASE)
            
            return transformed.strip()
        
        elif transformation_type == 'neutralize_roleplay':
            # Convert roleplay requests to direct questions
            transformed = re.sub(
                r'(?:pretend|act\s+like|roleplay\s+as)\s+you\s+are\s+([^.]*)[.]\s*(.+)',
                r'Can you provide information about \2',
                prompt,
                flags=re.IGNORECASE
            )
            return transformed
        
        elif transformation_type == 'clarify_intent':
            # Extract and clarify the real intent
            intent = self.extract_legitimate_intent(prompt)
            if intent:
                template = "I'm interested in learning about {}. Can you provide factual information?"
                # Extract topic from intent
                topic = re.sub(r'^(?:how|what|why|when|where|who|can|could|would|should|will)\s+', '', intent, flags=re.IGNORECASE)
                topic = re.sub(r'\?$', '', topic).strip()
                return template.format(topic)
            return prompt
        
        elif transformation_type == 'add_safety_context':
            return f"From an educational and safety-conscious perspective: {prompt}"
        
        elif transformation_type == 'redirect_to_safety':
            # Redirect harmful how-to requests to safety information
            if 'how to' in prompt.lower():
                topic = re.sub(r'.*how\s+to\s+', '', prompt, flags=re.IGNORECASE)
                topic = re.sub(r'[.?].*', '', topic).strip()
                return f"What are the safety considerations and legal aspects regarding {topic}?"
            return prompt
        
        elif transformation_type == 'educational_reframe':
            # Reframe as educational request
            return f"Can you provide educational information about the topic of: {self.extract_legitimate_intent(prompt)}"
        
        elif transformation_type == 'neutralize_bias':
            # Remove biased language and make neutral
            transformed = prompt
            bias_replacements = {
                r'all\s+(women|men)\s+are': r'some \1 might be',
                r'why\s+are\s+(women|men)\s+(better|worse)': r'what are different perspectives on \1 and',
                r'typical\s+(female|male)': r'some \1',
                r'naturally\s+(women|men)\s+are': r'\1 can be'
            }
            
            for pattern, replacement in bias_replacements.items():
                transformed = re.sub(pattern, replacement, transformed, flags=re.IGNORECASE)
            
            return transformed
        
        elif transformation_type == 'promote_equality':
            # Add equality-promoting context
            return f"From a perspective that values equality and avoids stereotypes: {prompt}"
        
        return prompt
    
    def _estimate_risk_reduction(self, original: str, transformed: str) -> float:
        """Estimate how much risk was reduced by the transformation."""
        # Simple heuristic based on removal of risk patterns
        original_risks = 0
        transformed_risks = 0
        
        for risk_type, config in self.risk_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, original.lower()):
                    original_risks += 1
                if re.search(pattern, transformed.lower()):
                    transformed_risks += 1
        
        if original_risks == 0:
            return 0.0
        
        reduction = (original_risks - transformed_risks) / original_risks
        return max(0.0, min(1.0, reduction))
    
    def generate_paraphrased_version(self, prompt: str) -> Optional[str]:
        """Generate a paraphrased version using a transformer model."""
        if not self.paraphraser_model:
            return None
        
        try:
            # Prepare input for T5 model
            input_text = f"paraphrase: {prompt}"
            inputs = self.paraphraser_tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.paraphraser_model.generate(
                    inputs,
                    max_length=200,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1
                )
            
            paraphrased = self.paraphraser_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return paraphrased
            
        except Exception as e:
            print(f"Paraphrasing failed: {e}")
            return None
    
    def update_prompt(self, prompt: str, strategy: str = "comprehensive") -> Dict:
        """
        Main method to update a potentially harmful prompt.
        
        Args:
            prompt: The original prompt to transform
            strategy: "rule_based", "paraphrasing", or "comprehensive"
        
        Returns:
            Dictionary containing transformation results
        """
        # Assess risk
        risk_level, detected_patterns = self.assess_prompt_risk(prompt)
        
        transformations = []
        
        # Apply rule-based transformations
        if strategy in ["rule_based", "comprehensive"]:
            rule_transformations = self.apply_rule_based_transformations(prompt, detected_patterns)
            transformations.extend(rule_transformations)
        
        # Apply paraphrasing if available and requested
        if strategy in ["paraphrasing", "comprehensive"] and self.paraphraser_model:
            paraphrased = self.generate_paraphrased_version(prompt)
            if paraphrased and paraphrased != prompt:
                # Assess risk reduction from paraphrasing
                _, paraphrased_patterns = self.assess_prompt_risk(paraphrased)
                risk_reduction = len(detected_patterns) - len(paraphrased_patterns)
                risk_reduction = max(0, risk_reduction) / max(1, len(detected_patterns))
                
                paraphrase_transformation = PromptTransformation(
                    original_prompt=prompt,
                    transformed_prompt=paraphrased,
                    transformation_type="paraphrasing",
                    risk_reduction=risk_reduction,
                    confidence=0.6,
                    explanation="Applied neural paraphrasing to reduce risk"
                )
                transformations.append(paraphrase_transformation)
        
        # Select best transformation
        best_transformation = None
        if transformations:
            # Sort by risk reduction, then by confidence
            transformations.sort(key=lambda x: (x.risk_reduction, x.confidence), reverse=True)
            best_transformation = transformations[0]
        
        result = {
            'original_prompt': prompt,
            'risk_assessment': {
                'risk_level': risk_level.value,
                'detected_patterns': detected_patterns
            },
            'transformations': [
                {
                    'transformed_prompt': t.transformed_prompt,
                    'transformation_type': t.transformation_type,
                    'risk_reduction': t.risk_reduction,
                    'confidence': t.confidence,
                    'explanation': t.explanation
                } for t in transformations
            ],
            'recommended_prompt': best_transformation.transformed_prompt if best_transformation else prompt,
            'risk_reduction_achieved': best_transformation.risk_reduction if best_transformation else 0.0
        }
        
        return result
    
    def batch_update_prompts(self, prompts: List[str], strategy: str = "comprehensive") -> List[Dict]:
        """Apply prompt updating to a batch of prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.update_prompt(prompt, strategy)
            result['prompt_id'] = i
            results.append(result)
        
        return results
    
    def analyze_transformation_effectiveness(self, results: List[Dict]) -> Dict:
        """Analyze the effectiveness of prompt transformations."""
        if not results:
            return {}
        
        analysis = {
            'total_prompts': len(results),
            'risk_level_distribution': {},
            'transformation_success': {},
            'risk_reduction_stats': {}
        }
        
        # Risk level distribution
        risk_levels = [r['risk_assessment']['risk_level'] for r in results]
        from collections import Counter
        analysis['risk_level_distribution'] = dict(Counter(risk_levels))
        
        # Transformation success rates
        successful_transformations = sum(1 for r in results if r['risk_reduction_achieved'] > 0)
        analysis['transformation_success'] = {
            'successful_transformations': successful_transformations,
            'success_rate': successful_transformations / len(results),
            'average_risk_reduction': sum(r['risk_reduction_achieved'] for r in results) / len(results)
        }
        
        # Risk reduction statistics
        risk_reductions = [r['risk_reduction_achieved'] for r in results]
        analysis['risk_reduction_stats'] = {
            'mean': sum(risk_reductions) / len(risk_reductions),
            'max': max(risk_reductions),
            'min': min(risk_reductions),
            'std': (sum((x - analysis['transformation_success']['average_risk_reduction'])**2 for x in risk_reductions) / len(risk_reductions))**0.5
        }
        
        return analysis

# Example usage functions
def create_safe_prompt_variants(original_prompts: List[str]) -> Dict[str, List[str]]:
    """Create safe variants of potentially harmful prompts."""
    updater = PromptUpdater()
    safe_variants = {}
    
    for prompt in original_prompts:
        result = updater.update_prompt(prompt)
        safe_variants[prompt] = [
            t['transformed_prompt'] for t in result['transformations']
        ]
    
    return safe_variants

def evaluate_prompt_safety_improvement(original_prompts: List[str], safety_classifier) -> Dict:
    """Evaluate how much safety is improved by prompt updating."""
    updater = PromptUpdater()
    improvements = []
    
    for prompt in original_prompts:
        # Get original safety score
        original_label, original_confidence = safety_classifier.predict(prompt)
        original_safety = original_confidence if original_label == 'safe' else (1 - original_confidence)
        
        # Transform prompt
        result = updater.update_prompt(prompt)
        updated_prompt = result['recommended_prompt']
        
        # Get updated safety score
        updated_label, updated_confidence = safety_classifier.predict(updated_prompt)
        updated_safety = updated_confidence if updated_label == 'safe' else (1 - updated_confidence)
        
        improvement = updated_safety - original_safety
        improvements.append({
            'original_prompt': prompt,
            'updated_prompt': updated_prompt,
            'original_safety_score': original_safety,
            'updated_safety_score': updated_safety,
            'improvement': improvement
        })
    
    return {
        'improvements': improvements,
        'average_improvement': sum(i['improvement'] for i in improvements) / len(improvements),
        'success_rate': sum(1 for i in improvements if i['improvement'] > 0) / len(improvements)
    }