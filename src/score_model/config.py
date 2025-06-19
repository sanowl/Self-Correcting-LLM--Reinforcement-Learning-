import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Enhanced configuration dataclass with environment variable support and validation.
    """
    # Core training parameters
    beta_1: float = field(default_factory=lambda: float(os.getenv('SCORE_BETA_1', '0.01')))
    beta_2: float = field(default_factory=lambda: float(os.getenv('SCORE_BETA_2', '0.1')))
    alpha: float = field(default_factory=lambda: float(os.getenv('SCORE_ALPHA', '5.0')))
    learning_rate: float = field(default_factory=lambda: float(os.getenv('SCORE_LEARNING_RATE', '1e-5')))
    batch_size: int = field(default_factory=lambda: int(os.getenv('SCORE_BATCH_SIZE', '1')))
    max_seq_len: int = field(default_factory=lambda: int(os.getenv('SCORE_MAX_SEQ_LEN', '1024')))
    num_epochs_stage_one: int = field(default_factory=lambda: int(os.getenv('SCORE_EPOCHS_STAGE_ONE', '1')))
    num_epochs_stage_two: int = field(default_factory=lambda: int(os.getenv('SCORE_EPOCHS_STAGE_TWO', '1')))
    
    # Device and reproducibility
    device: torch.device = field(default_factory=lambda: torch.device(
        os.getenv('SCORE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    ))
    seed: int = field(default_factory=lambda: int(os.getenv('SCORE_SEED', '42')))
    
    # Task and model configuration
    task: str = field(default_factory=lambda: os.getenv('SCORE_TASK', 'MATH'))
    model_variant: str = field(default_factory=lambda: os.getenv('SCORE_MODEL_VARIANT', 'decapoda-research/llama-7b-hf'))
    ablation: str = field(default_factory=lambda: os.getenv('SCORE_ABLATION', 'none'))
    
    # Paths
    data_path: str = field(default_factory=lambda: os.getenv('SCORE_DATA_PATH', './data'))
    output_dir: str = field(default_factory=lambda: os.getenv('SCORE_OUTPUT_DIR', './outputs'))
    log_dir: str = field(default_factory=lambda: os.getenv('SCORE_LOG_DIR', './logs'))
    model_cache_dir: Optional[str] = field(default_factory=lambda: os.getenv('SCORE_MODEL_CACHE_DIR'))
    
    # Training optimization
    num_workers: int = field(default_factory=lambda: int(os.getenv('SCORE_NUM_WORKERS', '2')))
    gradient_accumulation_steps: int = field(default_factory=lambda: int(os.getenv('SCORE_GRAD_ACCUM_STEPS', '1')))
    max_grad_norm: float = field(default_factory=lambda: float(os.getenv('SCORE_MAX_GRAD_NORM', '1.0')))
    warmup_steps: int = field(default_factory=lambda: int(os.getenv('SCORE_WARMUP_STEPS', '100')))
    
    # Monitoring and checkpointing
    save_steps: int = field(default_factory=lambda: int(os.getenv('SCORE_SAVE_STEPS', '1000')))
    logging_steps: int = field(default_factory=lambda: int(os.getenv('SCORE_LOGGING_STEPS', '10')))
    eval_steps: int = field(default_factory=lambda: int(os.getenv('SCORE_EVAL_STEPS', '1000')))
    max_eval_samples: int = field(default_factory=lambda: int(os.getenv('SCORE_MAX_EVAL_SAMPLES', '500')))
    save_total_limit: int = field(default_factory=lambda: int(os.getenv('SCORE_SAVE_TOTAL_LIMIT', '2')))
    
    # Performance and precision
    mixed_precision: bool = field(default_factory=lambda: os.getenv('SCORE_MIXED_PRECISION', 'false').lower() == 'true')
    dataloader_pin_memory: bool = field(default_factory=lambda: os.getenv('SCORE_PIN_MEMORY', 'true').lower() == 'true')
    gradient_checkpointing: bool = field(default_factory=lambda: os.getenv('SCORE_GRAD_CHECKPOINT', 'false').lower() == 'true')
    
    # Evaluation metrics
    compute_bleu: bool = field(default_factory=lambda: os.getenv('SCORE_COMPUTE_BLEU', 'true').lower() == 'true')
    compute_rouge: bool = field(default_factory=lambda: os.getenv('SCORE_COMPUTE_ROUGE', 'true').lower() == 'true')
    compute_cyclomatic_complexity: bool = field(default_factory=lambda: os.getenv('SCORE_COMPUTE_COMPLEXITY', 'true').lower() == 'true')
    
    # Health checks and monitoring
    enable_health_checks: bool = field(default_factory=lambda: os.getenv('SCORE_HEALTH_CHECKS', 'true').lower() == 'true')
    health_check_interval: int = field(default_factory=lambda: int(os.getenv('SCORE_HEALTH_CHECK_INTERVAL', '300')))
    enable_metrics: bool = field(default_factory=lambda: os.getenv('SCORE_ENABLE_METRICS', 'true').lower() == 'true')
    
    # Environment
    environment: str = field(default_factory=lambda: os.getenv('SCORE_ENV', 'development'))

    @classmethod
    def from_file(cls, config_file: str) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            Config: Configuration instance
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            logger.info(f"Loaded configuration from {config_file}")
            return cls(**config_data)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_file}: {e}")
            raise ValueError(f"Invalid configuration file format: {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            raise

    def save_to_file(self, config_file: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config_file (str): Path to save configuration file
        """
        try:
            config_dict = {}
            for field_name, field_def in self.__dataclass_fields__.items():
                value = getattr(self, field_name)
                if isinstance(value, torch.device):
                    config_dict[field_name] = str(value)
                else:
                    config_dict[field_name] = value
            
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
            raise

    def validate(self) -> None:
        """
        Comprehensive configuration validation.
        """
        errors = []
        
        # Validate numeric parameters
        if self.batch_size <= 0:
            errors.append("batch_size must be a positive integer")
        if self.max_seq_len <= 0:
            errors.append("max_seq_len must be a positive integer")
        if self.num_epochs_stage_one < 0:
            errors.append("num_epochs_stage_one must be non-negative")
        if self.num_epochs_stage_two < 0:
            errors.append("num_epochs_stage_two must be non-negative")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.num_workers < 0:
            errors.append("num_workers must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            errors.append("max_grad_norm must be positive")
        
        # Validate task
        if self.task not in ['MATH', 'CODE']:
            errors.append("task must be either 'MATH' or 'CODE'")
        
        # Validate environment
        if self.environment not in ['development', 'staging', 'production']:
            errors.append("environment must be one of: development, staging, production")
        
        # Validate and create directories
        for dir_attr, dir_name in [
            ('data_path', 'data directory'),
            ('output_dir', 'output directory'),
            ('log_dir', 'log directory')
        ]:
            dir_path = getattr(self, dir_attr)
            path_obj = Path(dir_path)
            
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created {dir_name} at {dir_path}")
                except Exception as e:
                    errors.append(f"Failed to create {dir_name} at {dir_path}: {e}")
            elif not path_obj.is_dir():
                errors.append(f"{dir_name} path exists but is not a directory: {dir_path}")
        
        # Validate model cache directory if specified
        if self.model_cache_dir:
            cache_path = Path(self.model_cache_dir)
            if not cache_path.exists():
                try:
                    cache_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created model cache directory at {self.model_cache_dir}")
                except Exception as e:
                    errors.append(f"Failed to create model cache directory: {e}")
        
        # Validate device
        if isinstance(self.device, str):
            try:
                self.device = torch.device(self.device)
            except Exception as e:
                errors.append(f"Invalid device specification: {self.device}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation successful")

    def get_model_save_path(self, step: Optional[int] = None) -> str:
        """
        Get the path for saving model checkpoints.
        
        Args:
            step (Optional[int]): Training step for checkpoint naming
            
        Returns:
            str: Model save path
        """
        if step is not None:
            return os.path.join(self.output_dir, f'checkpoint-{step}', 'model.bin')
        return os.path.join(self.output_dir, 'score_model.bin')

    def get_log_file_path(self, log_type: str = 'main') -> str:
        """
        Get the path for log files.
        
        Args:
            log_type (str): Type of log file
            
        Returns:
            str: Log file path
        """
        return os.path.join(self.log_dir, f'{log_type}.log')

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        config_dict = {}
        for field_name, _ in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, torch.device):
                config_dict[field_name] = str(value)
            else:
                config_dict[field_name] = value
        return config_dict

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(task={self.task}, model={self.model_variant}, env={self.environment})"

    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"Config({self.to_dict()})" 