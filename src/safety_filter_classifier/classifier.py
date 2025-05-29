
import torch
import torch.nn as nn
from transformers import (
    DistilBertModel, DistilBertTokenizer, 
    AutoModel, AutoTokenizer,
    TrainingArguments, Trainer
)
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle

class SafetyDataset(Dataset):
    """Dataset class for safety classification training."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SafetyClassifier(nn.Module):
    """
    Transformer-based safety classifier for detecting harmful content.
    """
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_classes: int = 3, dropout_rate: float = 0.3):
        super(SafetyClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained transformer
        self.transformer = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        # Label mapping
        self.label_to_id = {'safe': 0, 'biased_harmful': 1, 'unsafe_abusive': 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict safety label for a single text.
        Returns: (label, confidence)
        """
        self.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs['logits'], dim=-1)
            confidence, predicted_id = torch.max(probabilities, dim=-1)
            
            predicted_label = self.id_to_label[predicted_id.item()]
            
            return predicted_label, confidence.item()
    
    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Tuple[str, float]]:
        """Predict safety labels for a batch of texts."""
        self.eval()
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=encodings['input_ids'],
                    attention_mask=encodings['attention_mask']
                )
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs['logits'], dim=-1)
                confidences, predicted_ids = torch.max(probabilities, dim=-1)
                
                # Convert to labels
                for pred_id, conf in zip(predicted_ids.cpu().numpy(), confidences.cpu().numpy()):
                    label = self.id_to_label[pred_id]
                    results.append((label, conf))
        
        return results
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'label_to_id': self.label_to_id
        }, path)
    
    @classmethod
    def load_model(cls, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location='cpu')
        
        model = cls(
            model_name=checkpoint['model_name'],
            num_classes=checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.label_to_id = checkpoint['label_to_id']
        model.id_to_label = {v: k for k, v in model.label_to_id.items()}
        
        return model