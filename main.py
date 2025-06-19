import os
import argparse
import logging
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import nltk
import signal
import sys
from typing import Optional

from src.score_model.config import Config
from src.score_model.model import AdvancedModel
from src.score_model.dataset import BaseDataset
from src.score_model.trainer import SCoReTrainer
from src.score_model.utils import set_seed, load_json, load_model, setup_logging, cleanup_old_checkpoints

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Global variables for graceful shutdown
trainer_instance: Optional[SCoReTrainer] = None
structured_logger = None

def signal_handler(signum, frame):
    """Handle graceful shutdown."""
    if structured_logger:
        structured_logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    if trainer_instance:
        try:
            # Save current state
            trainer_instance.save_checkpoint(step=trainer_instance.global_step, is_final=True)
            structured_logger.info("Emergency checkpoint saved successfully")
        except Exception as e:
            if structured_logger:
                structured_logger.error(f"Failed to save emergency checkpoint: {e}")
    
    sys.exit(0)

def main():
    """
    Enhanced main function with comprehensive error handling and monitoring.
    """
    global trainer_instance, structured_logger
    
    parser = argparse.ArgumentParser(description="Enhanced SCoRe System")
    parser.add_argument('--task', type=str, default='MATH', choices=['MATH', 'CODE'])
    parser.add_argument('--model_variant', type=str, default='decapoda-research/llama-7b-hf')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--config_file', type=str, help='Path to configuration file')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--max_train_samples', type=int, default=1000, help='Maximum training samples')
    parser.add_argument('--max_eval_samples', type=int, default=100, help='Maximum evaluation samples')
    
    args = parser.parse_args()

    try:
        # Initialize configuration
        if args.config_file:
            config = Config.from_file(args.config_file)
        else:
            config = Config(
                task=args.task,
                model_variant=args.model_variant,
                data_path=args.data_path,
                output_dir=args.output_dir,
                mixed_precision=args.mixed_precision,
                gradient_checkpointing=args.gradient_checkpointing
            )

        # Validate configuration
        config.validate()
        
        # Setup logging
        structured_logger = setup_logging(config)
        structured_logger.info("Starting SCoRe training system", {
            'config': config.to_dict(),
            'args': vars(args)
        })

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Set seed for reproducibility
        set_seed(config.seed)

        # Load and validate data
        structured_logger.info("Loading training data...")
        train_file = os.path.join(config.data_path, 
            'math_train.json' if config.task == 'MATH' else 'mbpp_train.json')
        val_file = os.path.join(config.data_path,
            'math_test.json' if config.task == 'MATH' else 'HumanEval.jsonl')

        train_data = load_json(train_file, args.max_train_samples)
        val_data = load_json(val_file, args.max_eval_samples)
        
        if not train_data:
            raise ValueError(f"No training data found in {train_file}")
        if not val_data:
            raise ValueError(f"No validation data found in {val_file}")
        
        structured_logger.info(f"Data loaded successfully", {
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        })
        
        # Create datasets
        train_dataset = BaseDataset(train_data)
        val_dataset = BaseDataset(val_data)
        
        # Create data loaders with enhanced configuration
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers,
            pin_memory=config.dataloader_pin_memory,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            shuffle=False, 
            num_workers=config.num_workers,
            pin_memory=config.dataloader_pin_memory
        )

        # Initialize models with enhanced features
        structured_logger.info("Initializing models...")
        model = AdvancedModel(
            config.model_variant, 
            config.device,
            cache_dir=config.model_cache_dir,
            enable_gradient_checkpointing=config.gradient_checkpointing
        )
        
        ref_model = AdvancedModel(
            config.model_variant, 
            config.device,
            cache_dir=config.model_cache_dir
        )
        ref_model.model.eval()
        for param in ref_model.model.parameters():
            param.requires_grad = False

        # Log model information
        model_info = model.get_model_info()
        structured_logger.info("Model initialized successfully", model_info)

        # Setup optimizer with enhanced configuration
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
        
        # Calculate total steps and setup scheduler
        total_steps = len(train_loader) * (config.num_epochs_stage_one + config.num_epochs_stage_two)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

        # Initialize enhanced trainer
        trainer_instance = SCoReTrainer(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            logger=structured_logger
        )

        # Resume from checkpoint if specified
        if args.resume_from_checkpoint:
            structured_logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer_instance.load_checkpoint(args.resume_from_checkpoint)

        # Perform initial health checks
        structured_logger.info("Performing initial health checks...")
        model_health = model.health_check()
        if not model_health.is_healthy:
            raise RuntimeError(f"Model health check failed: {model_health.last_error}")
        
        ref_model_health = ref_model.health_check()
        if not ref_model_health.is_healthy:
            raise RuntimeError(f"Reference model health check failed: {ref_model_health.last_error}")
        
        structured_logger.info("Health checks passed successfully")

        # Train and evaluate
        structured_logger.info("Starting training process...")
        trainer_instance.train()
        
        structured_logger.info("Starting evaluation...")
        eval_results = trainer_instance.evaluate()
        
        structured_logger.info("Evaluation completed", eval_results)

        # Save final model
        model_save_path = config.get_model_save_path()
        torch.save(model.model.state_dict(), model_save_path)
        
        # Also save in HuggingFace format
        hf_save_path = os.path.join(config.output_dir, 'hf_model')
        model.save_pretrained(hf_save_path)
        
        structured_logger.info(f"Models saved successfully", {
            'pytorch_model': model_save_path,
            'huggingface_model': hf_save_path
        })
        
        # Save final configuration
        config_save_path = os.path.join(config.output_dir, 'config.json')
        config.save_to_file(config_save_path)
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(config.output_dir, config.save_total_limit)
        
        # Final health check and system metrics
        final_health = model.health_check()
        final_memory = model.get_memory_usage()
        
        structured_logger.info("Training completed successfully", {
            'final_health': final_health,
            'final_memory_usage': final_memory
        })

    except KeyboardInterrupt:
        if structured_logger:
            structured_logger.info("Training interrupted by user")
        else:
            print("Training interrupted by user")
    except Exception as e:
        error_msg = f"Training failed: {e}"
        if structured_logger:
            structured_logger.error(error_msg, {
                'error_type': type(e).__name__,
                'error_details': str(e)
            })
        else:
            logging.error(error_msg)
        raise

if __name__ == '__main__':
    main() 