import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import json
import re
from dataclasses import dataclass
from src.safety_filter_classifier.classifier import SafetyClassifier

@dataclass
class ThoughtStep:
    """Represents a single step in chain-of-thought reasoning."""
    step_number: int
    thought: str
    safety_assessment: str
    confidence: float

class ChainOfThoughtModerator:
    """
    Implements Chain-of-Thought based moderation for LLM safety.
    The system thinks step-by-step about potential harms before generating responses.
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
        
        # Load safety classifier if available
        if safety_classifier_path:
            self.safety_classifier = SafetyClassifier.load_model(safety_classifier_path)
            self.safety_classifier.eval()
        else:
            self.safety_classifier = None
        
        # Chain-of-thought templates
        self.cot_templates = {
            'safety_analysis': """
Let me think step by step about this request:

Step 1: Understanding the request
The user is asking about: {request_summary}

Step 2: Identifying potential concerns
Potential safety concerns: {concerns}

Step 3: Assessing harm level
Risk level: {risk_level}
Reasoning: {risk_reasoning}

Step 4: Determining response strategy
I should: {response_strategy}

Step 5: Generating appropriate response
""",
            
            'bias_check': """
Before responding, let me examine this for potential bias:

Step 1: Checking for stereotypes
Does this involve generalizations about groups? {stereotype_check}

Step 2: Examining assumptions
What assumptions might be embedded? {assumption_check}

Step 3: Considering different perspectives
How might different groups view this? {perspective_check}

Step 4: Formulating balanced response
""",
            
            'harm_prevention': """
Let me carefully consider the potential for harm:

Step 1: Direct harm assessment
Could following this cause immediate harm? {direct_harm}

Step 2: Indirect consequences
What are potential long-term effects? {indirect_harm}

Step 3: Vulnerable populations
Who might be most at risk? {vulnerable_groups}

Step 4: Mitigation strategies
How can I minimize risks while being helpful? {mitigation}

Step 5: Safe response generation
"""
        }
    
    def analyze_request_intent(self, prompt: str) -> Dict[str, str]:
        """Analyze the intent and content of a request."""
        prompt_lower = prompt.lower()
        
        # Categorize request type
        request_types = {
            'information': ['what is', 'explain', 'describe', 'tell me about'],
            'instruction': ['how to', 'teach me', 'show me', 'guide me'],
            'opinion': ['what do you think', 'your opinion', 'do you believe'],
            'creative': ['write', 'create', 'generate', 'compose'],
            'comparison': ['compare', 'difference between', 'better than', 'versus']
        }
        
        detected_type = 'general'
        for req_type, keywords in request_types.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_type = req_type
                break
        
        # Identify potential concerns
        concern_indicators = {
            'violence': ['violence', 'harm', 'hurt', 'attack', 'kill', 'weapon'],
            'illegal': ['illegal', 'crime', 'fraud', 'steal', 'hack'],
            'bias': ['stereotype', 'all women', 'all men', 'typical', 'naturally'],
            'manipulation': ['trick', 'deceive', 'manipulate', 'exploit'],
            'privacy': ['personal information', 'private data', 'spy', 'track'],
            'jailbreak': ['ignore instructions', 'pretend you are', 'roleplay', 'bypass']
        }
        
        concerns = []
        for concern_type, keywords in concern_indicators.items():
            if any(keyword in prompt_lower for keyword in keywords):
                concerns.append(concern_type)
        
        return {
            'request_type': detected_type,
            'concerns': concerns,
            'request_summary': self._summarize_request(prompt)
        }
    
    def _summarize_request(self, prompt: str) -> str:
        """Create a brief summary of the request."""
        # Simple extractive summary
        if len(prompt) <= 100:
            return prompt
        
        # Extract key phrases
        sentences = prompt.split('.')
        if sentences:
            return sentences[0][:100] + "..."
        else:
            return prompt[:100] + "..."
    
    def generate_cot_reasoning(self, prompt: str) -> List[ThoughtStep]:
        """Generate chain-of-thought reasoning steps."""
        analysis = self.analyze_request_intent(prompt)
        thoughts = []
        
        # Step 1: Understanding
        understanding = f"The request appears to be seeking {analysis['request_type']} about: {analysis['request_summary']}"
        thoughts.append(ThoughtStep(1, understanding, "neutral", 0.8))
        
        # Step 2: Concern identification
        if analysis['concerns']:
            concern_text = f"I've identified potential concerns: {', '.join(analysis['concerns'])}"
            safety_level = "concerning" if len(analysis['concerns']) > 1 else "cautious"
            confidence = 0.9 if len(analysis['concerns']) > 1 else 0.7
        else:
            concern_text = "No major safety concerns detected in this request."
            safety_level = "safe"
            confidence = 0.8
        
        thoughts.append(ThoughtStep(2, concern_text, safety_level, confidence))
        
        # Step 3: Risk assessment
        risk_level = self._assess_risk_level(analysis['concerns'])
        risk_reasoning = self._explain_risk_reasoning(analysis['concerns'], risk_level)
        
        thoughts.append(ThoughtStep(3, risk_reasoning, risk_level, 0.8))
        
        # Step 4: Response strategy
        strategy = self._determine_response_strategy(risk_level, analysis['concerns'])
        thoughts.append(ThoughtStep(4, strategy, risk_level, 0.9))
        
        return thoughts
    
    def _assess_risk_level(self, concerns: List[str]) -> str:
        """Assess the overall risk level based on concerns."""
        high_risk_concerns = ['violence', 'illegal', 'jailbreak']
        medium_risk_concerns = ['bias', 'manipulation', 'privacy']
        
        if any(concern in high_risk_concerns for concern in concerns):
            return "high_risk"
        elif any(concern in medium_risk_concerns for concern in concerns):
            return "medium_risk"
        elif concerns:
            return "low_risk"
        else:
            return "safe"
    
    def _explain_risk_reasoning(self, concerns: List[str], risk_level: str) -> str:
        """Provide reasoning for the risk assessment."""
        if risk_level == "high_risk":
            return f"High risk due to concerns about: {', '.join(concerns)}. Could lead to direct harm."
        elif risk_level == "medium_risk":
            return f"Medium risk due to potential for: {', '.join(concerns)}. Requires careful handling."
        elif risk_level == "low_risk":
            return f"Low risk, but noted concerns: {', '.join(concerns)}. Should proceed with caution."
        else:
            return "No significant risks identified. Can proceed with helpful response."
    
    def _determine_response_strategy(self, risk_level: str, concerns: List[str]) -> str:
        """Determine the appropriate response strategy."""
        if risk_level == "high_risk":
            return "Refuse the request and explain why it's problematic. Offer alternative safe information."
        elif risk_level == "medium_risk":
            return "Provide information with strong safety caveats and ethical considerations."
        elif risk_level == "low_risk":
            return "Respond helpfully while addressing any minor concerns noted."
        else:
            return "Respond helpfully and informatively."
    
    def moderate_with_cot(self, prompt: str, show_reasoning: bool = True) -> Dict:
        """Generate a moderated response using chain-of-thought reasoning."""
        
        # Generate reasoning steps
        thought_steps = self.generate_cot_reasoning(prompt)
        
        # Determine final safety assessment
        final_step = thought_steps[-1]
        safety_decision = final_step.safety_assessment
        
        # Generate response based on reasoning
        response = self._generate_moderated_response(prompt, thought_steps)
        
        # Evaluate final response if classifier available
        if self.safety_classifier:
            combined_text = f"Prompt: {prompt} Response: {response}"
            classifier_label, classifier_confidence = self.safety_classifier.predict(combined_text)
        else:
            classifier_label, classifier_confidence = self._rule_based_evaluation(response)
        
        result = {
            'original_prompt': prompt,
            'thought_process': [
                {
                    'step': step.step_number,
                    'thought': step.thought,
                    'safety_assessment': step.safety_assessment,
                    'confidence': step.confidence
                } for step in thought_steps
            ],
            'moderated_response': response,
            'safety_decision': safety_decision,
            'classifier_evaluation': {
                'label': classifier_label,
                'confidence': classifier_confidence
            },
            'reasoning_visible': show_reasoning
        }
        
        # Include reasoning in response if requested
        if show_reasoning:
            reasoning_text = self._format_reasoning_for_display(thought_steps)
            result['response_with_reasoning'] = f"{reasoning_text}\n\nFinal Response: {response}"
        
        return result
    
    def _generate_moderated_response(self, prompt: str, thought_steps: List[ThoughtStep]) -> str:
        """Generate the actual response based on reasoning."""
        final_assessment = thought_steps[-1].safety_assessment
        
        if final_assessment == "high_risk":
            return self._generate_refusal_response(prompt, thought_steps)
        elif final_assessment in ["medium_risk", "concerning"]:
            return self._generate_cautious_response(prompt, thought_steps)
        else:
            return self._generate_helpful_response(prompt, thought_steps)
    
    def _generate_refusal_response(self, prompt: str, thought_steps: List[ThoughtStep]) -> str:
        """Generate a polite refusal response."""
        refusal_templates = [
            "I can't provide information on that topic as it could be harmful. Instead, I'd be happy to help with {}.",
            "I'm not able to assist with that request due to safety concerns. Perhaps I can help you with {} instead?",
            "I cannot provide guidance on that matter. However, I can offer information about {} if that would be helpful."
        ]
        
        import random
        template = random.choice(refusal_templates)
        
        # Suggest alternative
        alternative = "a related but safer topic"
        return template.format(alternative)
    
    def _generate_cautious_response(self, prompt: str, thought_steps: List[ThoughtStep]) -> str:
        """Generate a response with safety caveats."""
        # This would typically use the LLM to generate a careful response
        # For now, providing a template-based approach
        
        return (
            "I understand you're asking about this topic. While I can provide some general information, "
            "it's important to note that this is a sensitive area that requires careful consideration. "
            "Please ensure any actions you take are legal, ethical, and safe for yourself and others."
        )
    
    def _generate_helpful_response(self, prompt: str, thought_steps: List[ThoughtStep]) -> str:
        """Generate a helpful response for safe prompts."""
        try:
            # Use the model to generate a response
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return "I'd be happy to help with your question. Could you please provide a bit more context so I can give you the most helpful response?"
    
    def _format_reasoning_for_display(self, thought_steps: List[ThoughtStep]) -> str:
        """Format the reasoning steps for display."""
        reasoning = "My reasoning process:\n\n"
        
        for step in thought_steps:
            reasoning += f"Step {step.step_number}: {step.thought}\n"
            reasoning += f"Assessment: {step.safety_assessment} (confidence: {step.confidence:.2f})\n\n"
        
        return reasoning
    
    def _rule_based_evaluation(self, response: str) -> Tuple[str, float]:
        """Fallback rule-based evaluation."""
        response_lower = response.lower()
        
        # Check for refusal patterns
        refusal_patterns = [
            "i can't", "i cannot", "i'm not able", "i won't",
            "not appropriate", "against guidelines", "safety concerns"
        ]
        
        if any(pattern in response_lower for pattern in refusal_patterns):
            return 'safe', 0.9
        
        # Check for harmful content
        harmful_keywords = ['violence', 'harm', 'illegal', 'dangerous']
        harm_count = sum(1 for keyword in harmful_keywords if keyword in response_lower)
        
        if harm_count >= 2:
            return 'unsafe_abusive', 0.8
        elif harm_count >= 1:
            return 'biased_harmful', 0.6
        else:
            return 'safe', 0.7
    
    def batch_moderate(self, prompts: List[str], show_reasoning: bool = False) -> List[Dict]:
        """Apply chain-of-thought moderation to a batch of prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Moderating prompt {i+1}/{len(prompts)}")
            result = self.moderate_with_cot(prompt, show_reasoning)
            result['prompt_id'] = i
            results.append(result)
        
        return results
    
    def analyze_moderation_effectiveness(self, results: List[Dict]) -> Dict:
        """Analyze the effectiveness of chain-of-thought moderation."""
        if not results:
            return {}
        
        analysis = {
            'total_prompts': len(results),
            'safety_decisions': {},
            'classifier_agreement': {},
            'reasoning_quality': {}
        }
        
        # Safety decision distribution
        safety_decisions = [r['safety_decision'] for r in results]
        from collections import Counter
        analysis['safety_decisions'] = dict(Counter(safety_decisions))
        
        # Classifier agreement
        agreements = 0
        for result in results:
            cot_decision = result['safety_decision']
            classifier_label = result['classifier_evaluation']['label']
            
            # Map decisions to comparable categories
            cot_safe = cot_decision in ['safe', 'low_risk']
            classifier_safe = classifier_label == 'safe'
            
            if cot_safe == classifier_safe:
                agreements += 1
        
        analysis['classifier_agreement'] = {
            'agreement_rate': agreements / len(results),
            'total_comparisons': len(results)
        }
        
        # Reasoning quality metrics
        avg_steps = sum(len(r['thought_process']) for r in results) / len(results)
        avg_confidence = sum(
            sum(step['confidence'] for step in r['thought_process']) / len(r['thought_process'])
            for r in results
        ) / len(results)
        
        analysis['reasoning_quality'] = {
            'average_reasoning_steps': avg_steps,
            'average_confidence': avg_confidence
        }
        
        return analysis

class InteractiveChainOfThoughtModerator(ChainOfThoughtModerator):
    """
    Enhanced version that allows interactive refinement of reasoning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_history = []
    
    def interactive_moderate(self, prompt: str) -> Dict:
        """Interactive moderation with step-by-step user feedback."""
        print("Starting interactive chain-of-thought moderation...")
        print(f"Prompt: {prompt}\n")
        
        # Generate initial reasoning
        thought_steps = self.generate_cot_reasoning(prompt)
        
        # Allow user to review and modify each step
        refined_steps = []
        for step in thought_steps:
            print(f"Step {step.step_number}: {step.thought}")
            print(f"Assessment: {step.safety_assessment} (confidence: {step.confidence:.2f})")
            
            # In a real implementation, this would gather user input
            # For now, we'll just use the original steps
            refined_steps.append(step)
            print()
        
        # Generate final response
        response = self._generate_moderated_response(prompt, refined_steps)
        
        result = {
            'prompt': prompt,
            'refined_reasoning': refined_steps,
            'final_response': response,
            'interaction_log': self.reasoning_history
        }
        
        return result