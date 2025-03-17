import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def train_epoch(model, data_loader, optimizer, scheduler, criterion, device, use_mixed_precision=False):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to use for training
        use_mixed_precision: Whether to use mixed precision training (only for CUDA)
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Use tqdm for progress bar
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on the validation or test set.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use for evaluation
        
    Returns:
        tuple: (average loss, accuracy, precision, recall, f1, auc)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Move to CPU for sklearn metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5  # Default value if AUC calculation fails
    
    return avg_loss, accuracy, precision, recall, f1, auc

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, 
                class_weights=None, warmup_steps=0, output_dir='models', use_mixed_precision=False):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs (int): Number of epochs to train for
        learning_rate (float): Learning rate for optimizer
        device: Device to use for training
        class_weights (torch.Tensor, optional): Class weights for loss function
        warmup_steps (int): Number of warmup steps for learning rate scheduler
        output_dir (str): Directory to save model checkpoints
        use_mixed_precision (bool): Whether to use mixed precision training (only for CUDA)
        
    Returns:
        dict: Dictionary containing training history
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Initialize learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize variables for early stopping
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, use_mixed_precision)
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        # Save model if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_auc': val_auc
            }, os.path.join(output_dir, 'best_model.pt'))
            
            print(f"Saved best model with F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    return history

def plot_training_history(history, output_dir='plots'):
    """
    Plot training history.
    
    Args:
        history (dict): Dictionary containing training history
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1')
    plt.plot(history['val_auc'], label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'metrics.png'))
    plt.close()

def plot_confusion_matrix(model, data_loader, device, output_dir='plots'):
    """
    Plot confusion matrix.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Move to CPU for sklearn metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Gaming', 'Gaming'],
                yticklabels=['Non-Gaming', 'Gaming'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close() 