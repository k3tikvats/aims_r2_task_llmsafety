# LLM Safety: Red Teaming and Mitigation

A comprehensive implementation of safety techniques for Large Language Models (LLMs), focusing on red teaming, safety classification, and mitigation strategies.

## 🎯 Project Overview

This project explores how Large Language Models can be prompted to generate unsafe, harmful, or biased outputs and implements techniques to detect and mitigate such behaviors. The project is structured in three main phases:

1. **Red Teaming**: Generate adversarial prompts and collect potentially harmful responses
2. **Safety Classification**: Train a classifier to detect unsafe outputs
3. **Mitigation**: Implement and evaluate various safety mitigation techniques

##  Project Structure

```
aims_r2_task_llmsafety/
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── data/                     # Generated datasets and results
├── models/                   # Trained models
├── results/                  # Evaluation results and reports
├── visualizations/          # Generated plots and charts
└── src/
    ├── red_teaming/
    │   ├── prompt_generator.py      # Adversarial prompt generation
    │   ├── response_collector.py    # LLM response collection
    │   └── labeling_system.py       # Safety labeling system
    ├── safety_filter_classifier/
    │   ├── classifier.py            # Safety classifier model
    │   ├── training.py              # Training pipeline
    │   └── evaluation.py            # Evaluation utilities
    ├── mitigation/
    │   ├── rejection_sampling.py    # Rejection sampling mitigation
    │   ├── chain_of_thought_moderator.py  # CoT-based moderation
    │   └── prompt_updater.py        # Adversarial prompt transformation
    └── utils/
        └── data_processing.py       # Data utilities and processing
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd aims_r2_task_llmsafety
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Complete Pipeline

Execute the entire pipeline:

```bash
python main.py --phase all --num_prompts 500
```

Or run individual phases:

```bash
# Red teaming only
python main.py --phase red_team --num_prompts 1000

# Training only (requires existing dataset)
python main.py --phase train

# Mitigation testing only (requires trained classifier)
python main.py --phase mitigate

# Generate report only
python main.py --phase report
```

## 📊 Detailed Components

### Part 1: Red Teaming

The red teaming module generates adversarial prompts designed to elicit unsafe responses:

- **Bias Prompts**: Target demographic stereotyping and unfair generalizations
- **Harm Prompts**: Seek instructions for harmful or dangerous activities
- **Jailbreak Prompts**: Attempt to bypass safety guidelines through roleplay or manipulation

Key features:

- 15+ adversarial prompt categories
- Automated response collection from open-source LLMs
- Rule-based and heuristic safety labeling
- Export capabilities for manual annotation

### Part 2: Safety Filter Implementation

A transformer-based classifier to detect unsafe LLM outputs:

- **Architecture**: DistilBERT-based classifier with 3 safety categories
- **Categories**: Safe, Biased/Harmful, Unsafe/Abusive
- **Training**: Automated training pipeline with validation and early stopping
- **Evaluation**: Comprehensive metrics including confusion matrix and per-class analysis

### Part 3: Mitigation Techniques

Four different mitigation strategies are implemented and evaluated:

#### 1. Rejection Sampling

- Generates multiple response candidates
- Selects the safest response using the safety classifier
- Adaptive version adjusts generation parameters based on prompt risk

#### 2. Chain-of-Thought Moderation

- Step-by-step reasoning about safety concerns
- Explicit assessment of potential harms
- Transparent decision-making process

#### 3. Prompt Updating/Transformation

- Identifies and neutralizes adversarial prompt patterns
- Rule-based transformations for common attack vectors
- Neural paraphrasing for sophisticated prompt rewriting

#### 4. Multi-Strategy Ensemble

- Combines multiple mitigation approaches
- Adaptive selection based on prompt characteristics
- Configurable safety thresholds

## 🔧 Configuration Options

### Model Selection

```python
# Different base models can be used
GENERATION_MODEL = "microsoft/DialoGPT-medium"  # or "gpt2", "facebook/blenderbot-400M-distill"
CLASSIFIER_MODEL = "distilbert-base-uncased"    # or "roberta-base", "bert-base-uncased"
```

### Safety Thresholds

```python
SAFETY_THRESHOLD = 0.7  # Minimum confidence for safe classification
MAX_GENERATION_ATTEMPTS = 10  # Maximum attempts in rejection sampling
```

## 📈 Results and Analysis

The project generates comprehensive evaluation reports including:

- **Dataset Statistics**: Distribution of prompt categories and safety labels
- **Classifier Performance**: Accuracy, precision, recall, F1 scores per class
- **Mitigation Effectiveness**: Success rates and safety improvements for each technique
- **Comparative Analysis**: Side-by-side evaluation of different mitigation strategies

Results are saved in multiple formats:

- JSON files for programmatic analysis
- Visualizations (confusion matrices, training curves, distribution plots)
- Markdown reports for human review

## 🛡️ Safety Considerations

This project is designed for research purposes to improve LLM safety. Key safety measures:

- **Responsible Disclosure**: Adversarial prompts use placeholders and safe substitutions
- **No Real Harm**: All examples avoid actual harmful content
- **Educational Focus**: Emphasis on understanding and prevention
- **Ethical Guidelines**: Following responsible AI research practices

## 🔬 Research Applications

This codebase supports research in:

- **Adversarial ML**: Understanding prompt-based attacks on LLMs
- **Safety Alignment**: Developing robust safety measures for AI systems
- **Red Teaming**: Systematic vulnerability assessment
- **Mitigation Strategies**: Comparative evaluation of safety techniques

## 📋 Usage Examples

### Basic Red Teaming

```python
from src.red_teaming.prompt_generator import AdversarialPromptGenerator
from src.red_teaming.response_collector import LLMResponseCollector

generator = AdversarialPromptGenerator()
prompts = generator.generate_bias_prompts(num_prompts=100)

collector = LLMResponseCollector()
responses = collector.collect_responses(prompts)
```

### Safety Classification

```python
from src.safety_filter_classifier.classifier import SafetyClassifier

classifier = SafetyClassifier.load_model('models/safety_classifier.pth')
label, confidence = classifier.predict("Your prompt here")
```

### Mitigation Application

```python
from src.mitigation.rejection_sampling import RejectionSampler

sampler = RejectionSampler(safety_classifier_path='models/safety_classifier.pth')
result = sampler.rejection_sample("Potentially harmful prompt")
safe_response = result['selected_response']
```


## 🙏 Acknowledgments

- Hugging Face Transformers library
- OpenAI's work on AI safety
- Research community contributions to responsible AI

