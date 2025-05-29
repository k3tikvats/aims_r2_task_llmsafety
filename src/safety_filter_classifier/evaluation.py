
import torch
import json
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from .classifier import SafetyClassifier

class SafetyClassifierEvaluator:
    """Evaluation utilities for the safety classifier."""
    
    def __init__(self, model: SafetyClassifier):
        self.model = model
    
    def evaluate_on_dataset(self, test_texts: List[str], test_labels: List[str]) -> Dict:
        """Evaluate model on a test dataset."""
        # Convert string labels to numeric
        numeric_labels = [self.model.label_to_id[label] for label in test_labels]
        
        # Get predictions
        predictions = self.model.batch_predict(test_texts)
        predicted_labels = [self.model.label_to_id[pred[0]] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(numeric_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            numeric_labels, predicted_labels, average='weighted'
        )
        
        # Per-class metrics
        per_class_metrics = classification_report(
            numeric_labels, predicted_labels,
            target_names=list(self.model.id_to_label.values()),
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_metrics': per_class_metrics,
            'predictions': predictions,
            'confidences': confidences
        }
    
    def analyze_errors(self, test_texts: List[str], test_labels: List[str]) -> Dict:
        """Analyze prediction errors."""
        predictions = self.model.batch_predict(test_texts)
        
        errors = []
        for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
            pred_label, confidence = predictions[i]
            
            if pred_label != true_label:
                errors.append({
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence
                })
        
        # Categorize errors
        false_positives = [e for e in errors if e['true_label'] == 'safe']
        false_negatives = [e for e in errors if e['predicted_label'] == 'safe']
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(test_texts),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'error_examples': errors[:10]  # Show first 10 errors
        }

# Example usage
if __name__ == "__main__":
    from .training import SafetyClassifierTrainer
    
    # Initialize model
    model = SafetyClassifier()
    
    # Initialize trainer
    trainer = SafetyClassifierTrainer(model)
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data('data/labeled_dataset.json')
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=5)
    
    # Evaluate model
    trainer.evaluate(test_loader)
    
    # Plot training history
    trainer.plot_training_history()
    
    print("Safety classifier training completed!")