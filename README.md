# SCoRe: Self-Correcting Language Model with Reinforcement Learning

This project implements a self-correcting language model that uses reinforcement learning to improve its outputs through multiple attempts.

## Features

- Two-stage training process with reinforcement learning
- Support for both mathematical and coding tasks
- Comprehensive evaluation metrics including BLEU, ROUGE, and cyclomatic complexity
- Mixed precision training support
- Modular and extensible architecture

## Installation

1. Clone the repository:
```bash
cd Self-Correcting-LLM--Reinforcement-Learning-
```

2. Install the package:
```bash
pip install -e .
```

## Usage

### Training

To train the model on mathematical tasks:
```bash
python main.py --task MATH --data_path ./data --output_dir ./outputs
```

To train on coding tasks:
```bash
python main.py --task CODE --data_path ./data --output_dir ./outputs
```

Additional options:
- `--model_variant`: Specify the model variant (default: 'decapoda-research/llama-7b-hf')
- `--mixed_precision`: Enable mixed precision training
- `--no_bleu`: Disable BLEU score computation
- `--no_rouge`: Disable ROUGE score computation
- `--no_cyclomatic`: Disable cyclomatic complexity computation

### Project Structure

```
.
├── main.py              # Main training script
├── setup.py            # Package setup file
├── src/
│   └── score_model/    # Main package directory
│       ├── __init__.py
│       ├── config.py   # Configuration classes
│       ├── model.py    # Model implementation
│       ├── dataset.py  # Dataset classes
│       ├── trainer.py  # Training logic
│       └── utils.py    # Utility functions
├── data/               # Data directory
└── outputs/            # Output directory
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Other dependencies listed in setup.py
