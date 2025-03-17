import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class GamingIntentDataset(Dataset):
    """Custom PyTorch Dataset for the gaming intent classification task."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of text inputs
            labels (list): List of corresponding labels (0 for non-gaming, 1 for gaming)
            tokenizer: Tokenizer to use for encoding the texts
            max_length (int): Maximum sequence length for padding/truncation
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Convert to PyTorch tensors
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(data_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load and preprocess data from CSV file.
    
    Args:
        data_path (str): Path to CSV file.
        test_size (float): Proportion of data to use for testing.
        val_size (float): Proportion of data to use for validation.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        dict: Dictionary containing train, validation, and test splits.
    """
    df = pd.read_csv(data_path)
    
    print(f"Data type of is_gaming_related column: {df['is_gaming_related'].dtype}")
    print(f"Unique values in is_gaming_related column: {df['is_gaming_related'].unique()}")
    
    # Handle different formats of boolean values
    if df['is_gaming_related'].dtype == 'object':
        df['is_gaming_related'] = df['is_gaming_related'].map({'TRUE': True, 'FALSE': False})
    
    df['is_gaming_related'] = df['is_gaming_related'].astype(int)
    
    texts = df['text'].tolist()
    labels = df['is_gaming_related'].tolist()
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=train_labels
    )
    
    return {
        'train': (train_texts, train_labels),
        'val': (val_texts, val_labels),
        'test': (test_texts, test_labels)
    }

def create_data_loaders(data_splits, tokenizer, batch_size=16, max_length=128, num_workers=0):
    """
    Create PyTorch DataLoaders for each data split.
    
    Args:
        data_splits (dict): Dictionary containing train, validation, and test data splits
        tokenizer: Tokenizer to use for encoding the texts
        batch_size (int): Batch size for DataLoaders
        max_length (int): Maximum sequence length for padding/truncation
        num_workers (int): Number of subprocesses to use for data loading
        
    Returns:
        dict: Dictionary containing DataLoaders for each split
    """
    loaders = {}
    
    for split_name, (texts, labels) in data_splits.items():
        # Create dataset
        dataset = GamingIntentDataset(texts, labels, tokenizer, max_length)
        
        # Create dataloader
        shuffle = split_name == 'train'  # Only shuffle the training data
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    return loaders

def get_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        labels (list): List of labels
        
    Returns:
        torch.Tensor: Tensor of class weights
    """
    # Count occurrences of each class
    class_counts = np.bincount(labels)
    
    # Calculate weights (inversely proportional to class frequency)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    return torch.tensor(class_weights, dtype=torch.float) 