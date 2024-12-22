import os
import argparse
import logging
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import nltk

from src.score_model.config import Config
from src.score_model.model import AdvancedModel
from src.score_model.dataset import BaseDataset
from src.score_model.trainer import SCoReTrainer
from src.score_model.utils import set_seed, load_json, load_model

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to parse arguments and initiate training and evaluation.
    """
    parser = argparse.ArgumentParser(description="Advanced SCoRe System")
    parser.add_argument('--task', type=str, default='MATH', choices=['MATH', 'CODE'])
    parser.add_argument('--model_variant', type=str, default='decapoda-research/llama-7b-hf')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--mixed_precision', action='store_true')
    args = parser.parse_args()

    # Initialize configuration
    config = Config(
        task=args.task,
        model_variant=args.model_variant,
        data_path=args.data_path,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision
    )

    try:
        config.validate()
        set_seed(config.seed)

        # Load data
        train_file = os.path.join(config.data_path, 
            'math_train.json' if config.task == 'MATH' else 'mbpp_train.json')
        val_file = os.path.join(config.data_path,
            'math_test.json' if config.task == 'MATH' else 'HumanEval.jsonl')

        train_data = load_json(train_file, 1000)
        val_data = load_json(val_file, 100)
        
        train_dataset = BaseDataset(train_data)
        val_dataset = BaseDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
            shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers)

        # Initialize models
        model = AdvancedModel(config.model_variant, config.device)
        ref_model = AdvancedModel(config.model_variant, config.device)
        ref_model.model.eval()
        for param in ref_model.model.parameters():
            param.requires_grad = False

        # Setup optimizer and scheduler
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        total_steps = len(train_loader) * (config.num_epochs_stage_one + config.num_epochs_stage_two)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

        # Initialize trainer
        trainer = SCoReTrainer(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )

        # Train and evaluate
        trainer.train()
        trainer.evaluate()

        # Save model
        model_save_path = os.path.join(config.output_dir, 'score_model.bin')
        torch.save(model.model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")

    except Exception as e:
        logger.critical(f"Error: {e}")
        raise

if __name__ == '__main__':
    main() 