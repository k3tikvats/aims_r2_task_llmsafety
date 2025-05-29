#!/usr/bin/env python3
"""
Main execution script for the LLM Safety Project.
This script orchestrates the entire pipeline from red teaming to mitigation.
"""

import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from src.red_teaming.prompt_generator import AdversarialPromptGenerator
from src.red_teaming.response_collector import LLMResponseCollector
from src.red_teaming.labeling_system import ResponseLabeler
from src.safety_filter_classifier.classifier import SafetyClassifier
from src.safety_filter_classifier.training import SafetyClassifierTrainer
from src.mitigation.rejection_sampling import RejectionSampler, AdaptiveRejectionSampler
from src.mitigation.chain_of_thought_moderator import ChainOfThoughtModerator
from src.mitigation.prompt_updater import PromptUpdater
from src.utils.data_processing import DataProcessor

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data',
        'models',
        'results',
        'visualizations',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def run_red_teaming(num_prompts=500, model_name='gpt-3.5-turbo'):
    """Execute the red teaming phase."""
    print("=" * 50)
    print("PHASE 1: RED TEAMING")
    print("=" * 50)
    
    # Generate adversarial prompts
    print("Generating adversarial prompts...")
    generator = AdversarialPromptGenerator()
    prompts = generator.generate_comprehensive_dataset(num_prompts)
    
    # Save generated prompts
    with open('data/adversarial_prompts.json', 'w', encoding='utf-8') as f:
        json.dump([{
            'text': p.text,
            'category': p.category,
            'subcategory': p.subcategory,
            'expected_harm_level': p.expected_harm_level,
            'techniques_used': p.techniques_used
        } for p in prompts], f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(prompts)} adversarial prompts")
      # Collect responses from LLM
    print("Collecting LLM responses...")
    collector = LLMResponseCollector(model_name=model_name)
    responses = collector.collect_responses(prompts)
    
    # Save responses
    with open('data/llm_responses.json', 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    
    print(f"Collected {len(responses)} responses")
    
    # Label responses for safety
    print("Labeling responses for safety...")
    labeler = ResponseLabeler()
    labeled_responses = labeler.batch_label_responses(responses)
    
    # Save labeled dataset
    with open('data/labeled_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(labeled_responses, f, indent=2, ensure_ascii=False)
    
    print(f"Labeled {len(labeled_responses)} responses")
    
    # Generate annotation guidelines
    guidelines = labeler.generate_annotation_guidelines()
    with open('data/annotation_guidelines.md', 'w', encoding='utf-8') as f:
        f.write(guidelines)
    
    # Export for manual annotation
    labeler.export_for_annotation(labeled_responses, 'data/manual_annotation.json')
    
    return labeled_responses

def run_safety_classifier_training(data_path='data/labeled_dataset.json', epochs=3, batch_size=16, learning_rate=2e-5):
    """Train the safety classifier."""
    print("=" * 50)
    print("PHASE 2: SAFETY CLASSIFIER TRAINING")
    print("=" * 50)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    raw_data = processor.load_dataset(data_path)
    processed_data = processor.preprocess_responses(raw_data)
    
    # Create balanced dataset
    balanced_data = processor.create_balanced_dataset(processed_data, max_per_class=1000)
    
    # Generate statistics
    stats = processor.generate_statistics(balanced_data)
    print("Dataset statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Label distribution: {stats['label_distribution']}")
    
    # Visualize dataset
    processor.visualize_dataset(balanced_data, 'visualizations/dataset_overview.png')
    
    # Export processed data
    processor.export_for_training(balanced_data, 'data/processed')
    
    # Initialize and train classifier
    print("Training safety classifier...")
    model = SafetyClassifier()
    trainer = SafetyClassifierTrainer(model)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = trainer.prepare_data(data_path)
      # Train the model
    trainer.train(train_loader, val_loader, epochs=epochs, 
                 learning_rate=learning_rate, batch_size=batch_size,
                 save_path='models/safety_classifier.pth')
    
    # Evaluate the model
    test_report = trainer.evaluate(test_loader)
    
    # Plot training history
    trainer.plot_training_history()
    plt.savefig('visualizations/training_history.png')
    
    # Save evaluation report
    with open('results/classifier_evaluation.json', 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print("Safety classifier training completed!")
    return model

def run_mitigation_techniques(classifier_path='models/safety_classifier.pth'):
    """Test and evaluate mitigation techniques."""
    print("=" * 50)
    print("PHASE 3: MITIGATION TECHNIQUES")
    print("=" * 50)
    
    # Load test prompts (adversarial ones)
    with open('data/adversarial_prompts.json', 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    
    test_prompts = [p['text'] for p in prompt_data[:50]]  # Use subset for testing
    
    mitigation_results = {}
    
    # 1. Rejection Sampling
    print("Testing Rejection Sampling...")
    rejection_sampler = RejectionSampler(safety_classifier_path=classifier_path)
    rejection_results = rejection_sampler.batch_rejection_sample(test_prompts)
    
    # Analyze effectiveness
    rejection_analysis = rejection_sampler.analyze_sampling_effectiveness(rejection_results)
    mitigation_results['rejection_sampling'] = rejection_analysis
    
    # Save results
    with open('results/rejection_sampling_results.json', 'w') as f:
        json.dump(rejection_results, f, indent=2, ensure_ascii=False)
    
    # 2. Adaptive Rejection Sampling
    print("Testing Adaptive Rejection Sampling...")
    adaptive_sampler = AdaptiveRejectionSampler(safety_classifier_path=classifier_path)
    adaptive_results = []
    
    for prompt in test_prompts[:20]:  # Smaller subset due to computational cost
        result = adaptive_sampler.adaptive_rejection_sample(prompt)
        adaptive_results.append(result)
    
    with open('results/adaptive_rejection_results.json', 'w') as f:
        json.dump(adaptive_results, f, indent=2, ensure_ascii=False)
    
    # 3. Chain-of-Thought Moderation
    print("Testing Chain-of-Thought Moderation...")
    cot_moderator = ChainOfThoughtModerator(safety_classifier_path=classifier_path)
    cot_results = cot_moderator.batch_moderate(test_prompts[:20], show_reasoning=True)
    
    # Analyze effectiveness
    cot_analysis = cot_moderator.analyze_moderation_effectiveness(cot_results)
    mitigation_results['chain_of_thought'] = cot_analysis
    
    with open('results/cot_moderation_results.json', 'w') as f:
        json.dump(cot_results, f, indent=2, ensure_ascii=False)
    
    # 4. Prompt Updating
    print("Testing Prompt Updating...")
    prompt_updater = PromptUpdater()
    update_results = prompt_updater.batch_update_prompts(test_prompts[:30])
    
    # Analyze effectiveness
    update_analysis = prompt_updater.analyze_transformation_effectiveness(update_results)
    mitigation_results['prompt_updating'] = update_analysis
    
    with open('results/prompt_update_results.json', 'w') as f:
        json.dump(update_results, f, indent=2, ensure_ascii=False)
    
    # Save overall mitigation analysis
    with open('results/mitigation_comparison.json', 'w') as f:
        json.dump(mitigation_results, f, indent=2)
    
    print("Mitigation techniques evaluation completed!")
    return mitigation_results

def create_comprehensive_report():
    """Create a comprehensive report of all results."""
    print("=" * 50)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 50)
    
    report = {
        'project_overview': {
            'title': 'LLM Safety: Red Teaming and Mitigation',
            'description': 'Comprehensive analysis of LLM safety vulnerabilities and mitigation strategies',
            'phases': ['red_teaming', 'safety_classification', 'mitigation_techniques']
        },
        'results_summary': {}
    }
    
    # Load results from various phases
    try:
        with open('results/classifier_evaluation.json', 'r') as f:
            report['results_summary']['classifier_performance'] = json.load(f)
    except FileNotFoundError:
        print("Classifier evaluation results not found")
    
    try:
        with open('results/mitigation_comparison.json', 'r') as f:
            report['results_summary']['mitigation_effectiveness'] = json.load(f)
    except FileNotFoundError:
        print("Mitigation comparison results not found")
    
    # Generate summary statistics
    try:
        processor = DataProcessor()
        data = processor.load_dataset('data/labeled_dataset.json')
        stats = processor.generate_statistics(data)
        report['dataset_statistics'] = stats
    except FileNotFoundError:
        print("Dataset not found for statistics")
    
    # Save comprehensive report
    with open('results/comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create markdown report
    create_markdown_report(report)
    
    print("Comprehensive report created!")

def create_markdown_report(report_data):
    """Create a markdown format report."""
    markdown_content = f"""# LLM Safety Project Report

## Project Overview
{report_data['project_overview']['description']}

## Dataset Statistics
"""
    
    if 'dataset_statistics' in report_data:
        stats = report_data['dataset_statistics']
        markdown_content += f"""
- **Total Samples**: {stats.get('total_samples', 'N/A')}
- **Label Distribution**: {stats.get('label_distribution', {})}
- **Response Length Stats**: {stats.get('response_length_stats', {})}
"""
    
    markdown_content += """
## Safety Classifier Performance
"""
    
    if 'classifier_performance' in report_data.get('results_summary', {}):
        perf = report_data['results_summary']['classifier_performance']
        markdown_content += f"""
- **Test Accuracy**: {perf.get('accuracy', 'N/A')}
- **Precision**: {perf.get('precision', 'N/A')}
- **Recall**: {perf.get('recall', 'N/A')}
- **F1 Score**: {perf.get('f1', 'N/A')}
"""
    
    markdown_content += """
## Mitigation Techniques Effectiveness
"""
    
    if 'mitigation_effectiveness' in report_data.get('results_summary', {}):
        mit = report_data['results_summary']['mitigation_effectiveness']
        
        for technique, results in mit.items():
            markdown_content += f"""
### {technique.replace('_', ' ').title()}
- Performance metrics and analysis
"""
    
    markdown_content += """
## Key Findings and Recommendations

1. **Red Teaming Results**: [Summary of vulnerabilities found]
2. **Classifier Effectiveness**: [Assessment of safety classification performance]
3. **Mitigation Success**: [Evaluation of different mitigation strategies]
4. **Recommendations**: [Best practices and future work]

## Visualizations
- Dataset overview: `visualizations/dataset_overview.png`
- Training history: `visualizations/training_history.png`
- Confusion matrix: `confusion_matrix.png`

## Files Generated
- Adversarial prompts: `data/adversarial_prompts.json`
- LLM responses: `data/llm_responses.json`
- Labeled dataset: `data/labeled_dataset.json`
- Trained classifier: `models/safety_classifier.pth`
- Results: `results/` directory
"""
    
    with open('README_RESULTS.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='LLM Safety Project Pipeline')
    parser.add_argument('--phase', choices=['all', 'red_team', 'train', 'mitigate', 'report'], 
                       default='all', help='Phase to execute')
    parser.add_argument('--num_prompts', type=int, default=500, 
                       help='Number of adversarial prompts to generate')
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip classifier training (use existing model)')
    parser.add_argument('--model_name', type=str, default='mistralai/Devstral-Small-2505',
                       help='LLM model name for response collection')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs for safety classifier')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training safety classifier')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for safety classifier training')
    
    args = parser.parse_args()
      # Setup project structure
    setup_directories()
    
    print("🚀 Starting LLM Safety Project Pipeline")
    print(f"Phase: {args.phase}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Model: {args.model_name}")
    
    try:
        if args.phase in ['all', 'red_team']:
            labeled_data = run_red_teaming(args.num_prompts, args.model_name)
        
        if args.phase in ['all', 'train'] and not args.skip_training:
            model = run_safety_classifier_training(
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate
            )
        
        if args.phase in ['all', 'mitigate']:
            classifier_path = 'models/safety_classifier.pth'
            if os.path.exists(classifier_path):
                mitigation_results = run_mitigation_techniques(classifier_path)
            else:
                print("Warning: No trained classifier found. Skipping mitigation phase.")
        
        if args.phase in ['all', 'report']:
            create_comprehensive_report()
        
        print("✅ Pipeline completed successfully!")
        print("📊 Check the 'results/' directory for detailed outputs")
        print("📈 Check the 'visualizations/' directory for plots")
        print("📝 Check 'README_RESULTS.md' for the comprehensive report")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
