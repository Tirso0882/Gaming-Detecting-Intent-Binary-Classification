# Gaming Intent Detection

This project implements a binary classifier to detect gaming-related intent in user text inputs. It uses DistilBERT, a lightweight transformer model, to classify whether a given text is related to gaming or not.

## Project Structure

```
📦detecting_gaming_related_intent
 ┣ 📂datasets
 ┣ 📂docs
 ┣ 📂notebooks
 ┃ ┣ 📜exploratory_data_analysis.ipynb
 ┃ ┗ 📜model_training.ipynb
 ┣ 📂output
 ┃ ┣ 📂exploratory_data_analysis
 ┣ 📂src
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜data_preprocessing.py
 ┃ ┣ 📜demo.py
 ┃ ┣ 📜inference.py
 ┃ ┣ 📜main.py
 ┃ ┣ 📜model.py
 ┃ ┗ 📜train.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜LICENSE
 ┣ 📜requirements.txt
 ┣ 📜technical_report.md
 ┗ 📜technical_report.pdf
```

## Installation

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model with default parameters:
```bash
python src/main.py
```

To customize training parameters:
```bash
python -m src.main --data_path datasets/gaming_intent_dataset.csv --output_dir output --num_epochs 10 --batch_size 32 --learning_rate 3e-5 --max_length 256 --dropout_rate 0.2 --test_size 0.2 --val_size 0.1 --use_class_weights --mixed_precision
```

#### Explanation of Parameters:
`--data_path`: Path to the dataset CSV file.\
`--output_dir`: Directory to save model checkpoints and plots.\
`--num_epochs`: Number of epochs to train for.\
`--batch_size`: Batch size for training and evaluation.\
`--learning_rate`: Learning rate for the optimiser.\
`--max_length`: Maximum sequence length for padding/truncation.\
`--dropout_rate`: Dropout rate for regularization.\
`--test_size`: Proportion of data to use for testing.\
`-val_size`: Proportion of training data to use for validation.\
`--use_class_weights`: Flag to indicate whether to use class weights for imbalanced datasets.\
`--mixed_precision`: Flag to enable mixed precision training (only for CUDA).

### Running the Demo

To run the Gradio demo for interactive testing:
```bash
python src/demo.py
```

To create a publicly shareable link:
```bash
python src/demo.py --share
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.