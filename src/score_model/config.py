import os
import torch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Configuration dataclass for training parameters.
    """
    beta_1: float = 0.01
    beta_2: float = 0.1
    alpha: float = 5.0
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_seq_len: int = 1024
    num_epochs_stage_one: int = 1
    num_epochs_stage_two: int = 1
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = 42
    task: str = 'MATH'
    model_variant: str = 'decapoda-research/llama-7b-hf'
    ablation: str = 'none'
    data_path: str = './data'
    output_dir: str = './outputs'
    num_workers: int = 2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 1000
    logging_steps: int = 10
    eval_steps: int = 1000
    max_eval_samples: int = 500
    mixed_precision: bool = False
    save_total_limit: int = 2
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_cyclomatic_complexity: bool = True

    def validate(self) -> None:
        """
        Validate configuration parameters.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")
        if self.num_epochs_stage_one < 0 or self.num_epochs_stage_two < 0:
            raise ValueError("Number of epochs must be non-negative.")
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        if not os.path.isdir(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Created output directory at {self.output_dir}.")
            except Exception as e:
                logger.error(f"Failed to create output directory: {e}")
                raise 