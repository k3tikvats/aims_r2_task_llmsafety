import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
from pathlib import Path

class DataProcessor:
    """
    Comprehensive data processing utilities for LLM safety analysis.
    Handles dataset creation, cleaning, analysis, and visualization.
    """
    
    def __init__(self):
        self.safety_keywords = {
            'unsafe': [
                'violence', 'harm', 'hurt', 'kill', 'destroy', 'attack', 'murder',
                'illegal', 'crime', 'fraud', 'scam', 'steal', 'hack', 'bomb',
                'weapon', 'drug', 'poison', 'suicide', 'abuse'
            ],
            'bias': [
                'stereotype', 'prejudice', 'discrimination', 'racist', 'sexist',
                'all women', 'all men', 'typical', 'naturally', 'inherently',
                'obviously better', 'obviously worse', 'can\'t understand'
            ],
            'refusal': [
                'cannot', 'can\'t', 'won\'t', 'shouldn\'t', 'not able',
                'not comfortable', 'against guidelines', 'not appropriate',
                'sorry', 'apologize', 'refuse'
            ]
        }
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_dataset(self, data: List[Dict], file_path: str):
        """Save dataset to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        text = re.sub(r'[^\w\s\.,!?;:-]', '', text)  # Remove special chars
        
        return text
    
    def preprocess_responses(self, responses: List[Dict]) -> List[Dict]:
        """Preprocess response data for analysis."""
        processed = []
        
        for response in responses:
            # Clean text fields
            clean_response = response.copy()
            clean_response['prompt_text'] = self.clean_text(response.get('prompt_text', ''))
            clean_response['response_text'] = self.clean_text(response.get('response_text', ''))
            
            # Add computed features
            clean_response['response_length'] = len(clean_response['response_text'])
            clean_response['prompt_length'] = len(clean_response['prompt_text'])
            clean_response['word_count'] = len(clean_response['response_text'].split())
            
            # Add keyword analysis
            clean_response['keyword_analysis'] = self.analyze_keywords(
                clean_response['response_text']
            )
            
            processed.append(clean_response)
        
        return processed
    
    def analyze_keywords(self, text: str) -> Dict[str, int]:
        """Analyze presence of safety-related keywords."""
        text_lower = text.lower()
        analysis = {}
        
        for category, keywords in self.safety_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            analysis[f'{category}_keyword_count'] = count
        
        return analysis
    
    def create_balanced_dataset(self, data: List[Dict], max_per_class: int = 1000) -> List[Dict]:
        """Create a balanced dataset by sampling from each class."""
        # Group by safety label
        class_groups = {}
        for item in data:
            label = item.get('safety_label', 'unknown')
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(item)
        
        # Sample from each class
        balanced_data = []
        for label, items in class_groups.items():
            if len(items) > max_per_class:
                # Random sampling
                sampled = np.random.choice(items, max_per_class, replace=False)
                balanced_data.extend(sampled)
            else:
                balanced_data.extend(items)
        
        # Shuffle the dataset
        np.random.shuffle(balanced_data)
        return balanced_data
    
    def split_dataset(self, data: List[Dict], 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train, validation, and test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # Shuffle data
        shuffled_data = data.copy()
        np.random.shuffle(shuffled_data)
        
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def generate_statistics(self, data: List[Dict]) -> Dict:
        """Generate comprehensive statistics about the dataset."""
        if not data:
            return {}
        
        stats = {
            'total_samples': len(data),
            'label_distribution': {},
            'prompt_categories': {},
            'response_length_stats': {},
            'keyword_analysis': {}
        }
        
        # Label distribution
        labels = [item.get('safety_label', 'unknown') for item in data]
        stats['label_distribution'] = dict(Counter(labels))
        
        # Prompt categories
        categories = [item.get('prompt_category', 'unknown') for item in data]
        stats['prompt_categories'] = dict(Counter(categories))
        
        # Response length statistics
        lengths = [len(item.get('response_text', '')) for item in data]
        if lengths:
            stats['response_length_stats'] = {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths)
            }
        
        # Keyword analysis
        for category in self.safety_keywords.keys():
            counts = []
            for item in data:
                keyword_analysis = item.get('keyword_analysis', {})
                count = keyword_analysis.get(f'{category}_keyword_count', 0)
                counts.append(count)
            
            if counts:
                stats['keyword_analysis'][category] = {
                    'mean': np.mean(counts),
                    'total': np.sum(counts),
                    'samples_with_keywords': sum(1 for c in counts if c > 0)
                }
        
        return stats
    
    def visualize_dataset(self, data: List[Dict], save_path: str = None):
        """Create visualizations of the dataset."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Label distribution
        labels = [item.get('safety_label', 'unknown') for item in data]
        label_counts = Counter(labels)
        
        axes[0, 0].bar(label_counts.keys(), label_counts.values())
        axes[0, 0].set_title('Safety Label Distribution')
        axes[0, 0].set_xlabel('Safety Label')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Response length distribution
        lengths = [len(item.get('response_text', '')) for item in data]
        axes[0, 1].hist(lengths, bins=30, alpha=0.7)
        axes[0, 1].set_title('Response Length Distribution')
        axes[0, 1].set_xlabel('Response Length (characters)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Prompt categories
        categories = [item.get('prompt_category', 'unknown') for item in data]
        category_counts = Counter(categories)
        
        axes[1, 0].pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Prompt Category Distribution')
        
        # Keyword analysis heatmap
        keyword_data = []
        for category in self.safety_keywords.keys():
            category_counts = []
            for label in ['safe', 'biased_harmful', 'unsafe_abusive']:
                count = 0
                total = 0
                for item in data:
                    if item.get('safety_label') == label:
                        keyword_analysis = item.get('keyword_analysis', {})
                        count += keyword_analysis.get(f'{category}_keyword_count', 0)
                        total += 1
                
                avg_count = count / total if total > 0 else 0
                category_counts.append(avg_count)
            
            keyword_data.append(category_counts)
        
        if keyword_data:
            sns.heatmap(keyword_data, 
                       xticklabels=['safe', 'biased_harmful', 'unsafe_abusive'],
                       yticklabels=list(self.safety_keywords.keys()),
                       annot=True, fmt='.2f', ax=axes[1, 1])
            axes[1, 1].set_title('Average Keyword Count by Safety Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_for_training(self, data: List[Dict], output_dir: str):
        """Export data in formats suitable for training."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export as CSV for easy analysis
        df_data = []
        for item in data:
            row = {
                'prompt_id': item.get('prompt_id', ''),
                'prompt_text': item.get('prompt_text', ''),
                'response_text': item.get('response_text', ''),
                'safety_label': item.get('safety_label', ''),
                'prompt_category': item.get('prompt_category', ''),
                'response_length': len(item.get('response_text', '')),
                'word_count': len(item.get('response_text', '').split())
            }
            
            # Add keyword analysis
            keyword_analysis = item.get('keyword_analysis', {})
            for key, value in keyword_analysis.items():
                row[key] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path / 'dataset.csv', index=False)
        
        # Export training texts and labels separately
        texts = [f"Prompt: {item['prompt_text']} Response: {item['response_text']}" 
                for item in data]
        labels = [item.get('safety_label', 'safe') for item in data]
        
        with open(output_path / 'texts.json', 'w', encoding='utf-8') as f:
            json.dump(texts, f, indent=2, ensure_ascii=False)
        
        with open(output_path / 'labels.json', 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
        
        print(f"Data exported to {output_dir}")
        print(f"Total samples: {len(data)}")
        print(f"Label distribution: {dict(Counter(labels))}")

# Utility functions
def merge_datasets(*datasets: List[Dict]) -> List[Dict]:
    """Merge multiple datasets."""
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    return merged

def filter_by_confidence(data: List[Dict], min_confidence: float = 0.7) -> List[Dict]:
    """Filter data by label confidence."""
    return [item for item in data 
            if item.get('label_confidence', 0) >= min_confidence]

def augment_data(data: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
    """Simple data augmentation by paraphrasing (placeholder for more sophisticated methods)."""
    augmented = data.copy()
    
    # Simple augmentation: add slight variations
    for _ in range(augmentation_factor - 1):
        for item in data:
            if item.get('safety_label') in ['biased_harmful', 'unsafe_abusive']:
                # Create variation by adding context
                augmented_item = item.copy()
                augmented_item['prompt_id'] = f"{item['prompt_id']}_aug"
                augmented_item['prompt_text'] = f"Context: Please be careful. {item['prompt_text']}"
                augmented.append(augmented_item)
    
    return augmented