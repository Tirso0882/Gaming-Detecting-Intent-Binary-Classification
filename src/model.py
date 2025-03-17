import torch.nn as nn
from transformers import DistilBertModel


class GamingIntentClassifier(nn.Module):
    """DistilBERT-based model for gaming intent classification."""
    
    def __init__(self, num_classes=2, dropout_rate=0.1, pretrained_model_name="distilbert-base-uncased"):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes (2 for binary classification)
            dropout_rate (float): Dropout rate for regularization
            pretrained_model_name (str): Name of the pretrained DistilBERT model to use
        """
        super(GamingIntentClassifier, self).__init__()
        
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)
        
        config = self.distilbert.config
        hidden_size = config.hidden_size
        print(f"Hidden size from config: {hidden_size}")
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Logits for each class
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_output)
        
        return logits

def initialize_model(device, num_classes=2, dropout_rate=0.1, pretrained_model_name="distilbert-base-uncased"):
    """
    Initialize the model and move it to the specified device.
    
    Args:
        device (torch.device): Device to move the model to
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        pretrained_model_name (str): Name of the pretrained DistilBERT model to use
        
    Returns:
        GamingIntentClassifier: Initialized model
    """
    model = GamingIntentClassifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained_model_name=pretrained_model_name
    )
    
    model = model.to(device)
    
    return model 