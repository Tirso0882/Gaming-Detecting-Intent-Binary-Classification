import argparse
import os
import time

import numpy as np
import torch
from transformers import DistilBertTokenizer

from src.data_preprocessing import (create_data_loaders, get_class_weights,
                                    load_and_preprocess_data)
from src.model import initialize_model
from src.train import (evaluate, plot_confusion_matrix, plot_training_history,
                       train_model)


def main(args):
    """
    Main function to run the training process.
    
    Args:
        args: Command-line arguments
    """
    # Set device - optimized for M3 Pro
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using Apple Silicon MPS device: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU device: {device}")
    
    print(f"PyTorch version: {torch.__version__}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("Loading and preprocessing data...")
    data_splits = load_and_preprocess_data(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    
    # Create data loaders with optimized settings for M3 Pro
    print("Creating data loaders...")
    # Increase batch size for M3 Pro if using MPS
    batch_size = args.batch_size
    if device.type == 'mps' and batch_size < 32:
        batch_size = 32
        print(f"Increased batch size to {batch_size} for better M3 Pro performance")
    
    # Set num_workers for DataLoader to utilize M3 Pro's multiple cores
    num_workers = os.cpu_count() if device.type == 'mps' else 0
    
    data_loaders = create_data_loaders(
        data_splits,
        tokenizer,
        batch_size=batch_size,
        max_length=args.max_length,
        num_workers=num_workers
    )
    
    # Calculate class weights for imbalanced dataset
    if args.use_class_weights:
        print("Calculating class weights...")
        class_weights = get_class_weights(data_splits['train'][1])
        print(f"Class weights: {class_weights}")
        
        # Convert class weights to tensor and move to device
        if isinstance(class_weights, torch.Tensor):
            class_weights = class_weights.clone().detach().to(device)
        else:
            class_weights = torch.FloatTensor(class_weights).to(device)
    else:
        class_weights = None
    
    print("Initializing model...")
    model = initialize_model(
        device,
        num_classes=2,
        dropout_rate=args.dropout_rate,
        pretrained_model_name=args.model_name
    )
    
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size:,} parameters")
    
    # Enable mixed precision training for better performance on M3 Pro
    use_mixed_precision = args.mixed_precision and device.type == 'cuda'
    if use_mixed_precision:
        print("Using mixed precision training")
    
    start_time = time.time()
    
    print("Training model...")
    history = train_model(
        model,
        data_loaders['train'],
        data_loaders['val'],
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        class_weights=class_weights,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        use_mixed_precision=use_mixed_precision
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    print("Plotting training history...")
    plot_training_history(history, output_dir=args.output_dir)
    
    print("Loading best model for evaluation...")
    try:
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), 
                               map_location=device, weights_only=False)
    except RuntimeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), 
                               map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with validation F1: {checkpoint['val_f1']:.4f}")
    
    print("Evaluating on test set...")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate(
        model, data_loaders['test'], criterion, device
    )
    
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
          f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    print("Plotting confusion matrix...")
    plot_confusion_matrix(model, data_loaders['test'], device, output_dir=args.output_dir)
    
    with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
        f.write(f"Device: {device}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
    
    print(f"Results saved to {args.output_dir}")
    
    print("Benchmarking inference speed...")
    sample_text = "What's the best strategy for defeating the final boss in Elden Ring?"
    
    with torch.no_grad():
        encoding = tokenizer(
            sample_text,
            add_special_tokens=True,
            max_length=args.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
    print(f"Average inference time on {device}: {avg_time:.2f} ms")
    
    with open(os.path.join(args.output_dir, 'benchmark.txt'), 'w') as f:
        f.write(f"Average inference time on {device}: {avg_time:.2f} ms\n")
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Model size: {model_size:,} parameters\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a gaming intent classifier")
    
    parser.add_argument("--data_path", type=str, default="datasets/gaming_intent_dataset.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Proportion of training data to use for validation")
    
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Name of the pretrained DistilBERT model to use")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length for padding/truncation")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="Dropout rate for regularization")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Whether to use class weights for imbalanced dataset")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Whether to use mixed precision training (only for CUDA)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save model checkpoints and plots")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 