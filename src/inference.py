import pickle
import time

import numpy as np
import torch
import torch.serialization
from transformers import DistilBertTokenizer

from .model import GamingIntentClassifier


class GamingIntentPredictor:
    """Class for making predictions with a trained gaming intent classifier."""
    
    def __init__(self, model_path, device=None, tokenizer_name="distilbert-base-uncased"):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the saved model checkpoint
            device (torch.device, optional): Device to use for inference
            tokenizer_name (str): Name of the tokenizer to use
        """
        # Set device - optimized for MacBoolk Pro M3 Pro
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"Using Apple Silicon MPS device: {self.device}")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using CUDA device: {self.device}")
            else:
                self.device = torch.device('cpu')
                print(f"Using CPU device: {self.device}")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Cache for tokenized inputs to improve performance
        self.cache = {}
        
    def _load_model(self, model_path):
        """
        Load the model from a checkpoint.
        
        Args:
            model_path (str): Path to the saved model checkpoint
            
        Returns:
            GamingIntentClassifier: Loaded model
        """
        model = GamingIntentClassifier()
        
        print(f"Loading model from {model_path}...")
        
        try:
            from numpy._core.multiarray import scalar
            torch.serialization.add_safe_globals([scalar])
            print("Added NumPy scalar to safe globals")
        except (ImportError, AttributeError):
            print("Could not add NumPy scalar to safe globals, will try loading with weights_only=False")
        
        try:
            # First try with weights_only=False
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print("Successfully loaded checkpoint with weights_only=False")
        except Exception as e:
            print(f"Error loading with weights_only=False: {e}")
            try:
                # Then try with default settings
                checkpoint = torch.load(model_path, map_location=self.device)
                print("Successfully loaded checkpoint with default settings")
            except Exception as e:
                print(f"Error loading with default settings: {e}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return model
    
    def predict(self, text, return_probabilities=False):
        """
        Make a prediction for a single text input.
        
        Args:
            text (str): Text input to classify
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            int or tuple: Predicted class (0 for non-gaming, 1 for gaming) or tuple of (prediction, probabilities)
        """
        if text in self.cache:
            input_ids, attention_mask = self.cache[text]
        else:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            self.cache[text] = (input_ids, attention_mask)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
        
        if return_probabilities:
            if self.device.type == 'mps':
                # For MPS device, move to CPU first
                return prediction, probabilities.cpu().numpy()[0]
            else:
                return prediction, probabilities.cpu().numpy()[0]
        else:
            return prediction
    
    def benchmark(self, text, num_runs=100):
        """
        Benchmark inference speed.
        
        Args:
            text (str): Text input to use for benchmarking
            num_runs (int): Number of inference runs to perform
            
        Returns:
            float: Average inference time in milliseconds
        """
        self.predict(text)
        
        start_time = time.time()
        for _ in range(num_runs):
            self.predict(text)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time * 1000
    
    def clear_cache(self):
        """Clear the tokenization cache."""
        self.cache = {} 