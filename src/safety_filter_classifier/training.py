
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
# import wandb
from tqdm import tqdm
from .classifier import SafetyClassifier, SafetyDataset

class SafetyClassifierTrainer:
    """Trainer class for the safety classifier."""
    
    def __init__(self, model: SafetyClassifier, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def prepare_data(self, data_path: str, test_size: float = 0.2, val_size: float = 0.1):
        """Load and prepare training data."""
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract texts and labels
        texts = []
        labels = []
        
        for item in data:
            # Combine prompt and response for context
            text = f"Prompt: {item['prompt_text']} Response: {item['response_text']}"
            texts.append(text)
            
            # Convert safety label to numeric
            label = self.model.label_to_id[item['safety_label']]
            labels.append(label)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        # Create datasets
        train_dataset = SafetyDataset(X_train, y_train, self.model.tokenizer)
        val_dataset = SafetyDataset(X_val, y_val, self.model.tokenizer)
        test_dataset = SafetyDataset(X_test, y_test, self.model.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs['logits'], dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, epochs: int = 5, lr: float = 2e-5, save_path: str = 'safety_classifier.pth'):
        """Train the safety classifier."""
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        
        # Initialize wandb for logging (optional)
        # wandb.init(project="safety-classifier", config={"lr": lr, "epochs": epochs})
        
        best_val_accuracy = 0
        patience = 2
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log to wandb
            # wandb.log({
            #     "train_loss": train_loss,
            #     "train_accuracy": train_acc,
            #     "val_loss": val_loss,
            #     "val_accuracy": val_acc,
            #     "epoch": epoch
            # })
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                self.model.save_model(save_path)
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data."""
        test_loss, test_acc, test_preds, test_labels = self.validate(test_loader)
        
        # Classification report
        target_names = list(self.model.id_to_label.values())
        report = classification_report(
            test_labels, test_preds, 
            target_names=target_names, 
            output_dict=True
        )
        
        print("Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return report
    
    def plot_training_history(self):
        """Plot training history."""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()