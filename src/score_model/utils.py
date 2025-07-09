import json
import torch
import logging
import numpy as np
import time
import os
import psutil
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import logging.handlers
from .model import AdvancedModel
import secrets

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_mb: float
    gpu_utilization: float
    disk_usage_percent: float

@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    grad_norm: float
    batch_size: int
    tokens_per_second: float
    memory_usage_mb: float
    timestamp: float

class StructuredLogger:
    """Structured logging with metrics collection."""
    
    def __init__(self, name: str, log_dir: str, level: int = logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup structured logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics_file = self.log_dir / f"{name}_metrics.jsonl"
        
    def log_metrics(self, metrics: Union[SystemMetrics, TrainingMetrics, Dict[str, Any]]) -> None:
        """Log metrics to structured format."""
        try:
            if isinstance(metrics, (SystemMetrics, TrainingMetrics)):
                metrics_dict = asdict(metrics)
            else:
                metrics_dict = metrics
                
            metrics_dict['logger'] = self.name
            
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_dict) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message with optional structured data."""
        if extra:
            self.log_metrics(extra)
        self.logger.info(message)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with optional structured data."""
        if extra:
            self.log_metrics(extra)
        self.logger.error(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message with optional structured data."""
        if extra:
            self.log_metrics(extra)
        self.logger.warning(message)

def setup_logging(config) -> StructuredLogger:
    """Setup comprehensive logging system."""
    structured_logger = StructuredLogger(
        name='score_training',
        log_dir=config.log_dir,
        level=logging.DEBUG if config.environment == 'development' else logging.INFO
    )
    
    # Log system info at startup
    system_info = get_system_info()
    structured_logger.info("System information collected", system_info)
    
    return structured_logger

def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility with enhanced error handling.

    Args:
        seed (int): The seed value to set.
    """
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        secrets.SystemRandom().seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Additional CUDA determinism
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Seed set to {seed} with deterministic behavior enabled")
    except Exception as e:
        logger.error(f"Error setting seed: {e}")
        raise RuntimeError("Failed to set seed") from e

def load_json(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Enhanced JSON/JSONL loader with validation and error handling.

    Args:
        file_path (str): Path to the JSON or JSONL file.
        max_samples (Optional[int]): Maximum number of samples to load.

    Returns:
        List[Dict[str, Any]]: Loaded data.
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be a non-negative integer or None")

    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = []
    start_time = time.time()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == '.jsonl':
                for idx, line in enumerate(f):
                    if max_samples is not None and idx >= max_samples:
                        break
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {idx + 1}: {e}")
                            continue
            else:
                file_content = f.read().strip()
                if file_content:
                    try:
                        loaded_data = json.loads(file_content)
                        if isinstance(loaded_data, list):
                            data = loaded_data[:max_samples] if max_samples else loaded_data
                        else:
                            data = [loaded_data]
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in file {file_path}: {e}")
                        raise ValueError(f"Invalid JSON format in file: {file_path}") from e
                else:
                    logger.info(f"Empty file: {file_path}")
                    
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        logger.error(f"Unexpected error while loading JSON from {file_path}: {e}")
        raise RuntimeError(f"Failed to load data from {file_path}") from e

    load_time = time.time() - start_time
    logger.info(f"Loaded {len(data)} samples from {file_path} in {load_time:.2f}s")
    return data

def load_model(model_path: str, model_variant: str, device: torch.device, **kwargs) -> AdvancedModel:
    """
    Enhanced model loader with validation and error handling.

    Args:
        model_path (str): Path to the saved model state dict.
        model_variant (str): Model variant identifier.
        device (torch.device): Device to load the model onto.
        **kwargs: Additional arguments for AdvancedModel.

    Returns:
        AdvancedModel: Loaded model instance.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        start_time = time.time()
        
        # Initialize model
        advanced_model = AdvancedModel(model_variant, device, **kwargs)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        advanced_model.model.load_state_dict(state_dict)
        advanced_model.model.to(device)
        advanced_model.model.eval()
        
        load_time = time.time() - start_time
        
        # Verify model health
        health = advanced_model.health_check()
        if not health.is_healthy:
            raise RuntimeError(f"Model health check failed: {health.last_error}")
        
        logger.info(f"Model loaded from {model_path} in {load_time:.2f}s and health check passed")
        return advanced_model
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}") from e

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dict[str, Any]: System information
    """
    info = {
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'platform': os.name,
        'cpu_count': os.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        })
    
    return info

def collect_system_metrics() -> SystemMetrics:
    """
    Collect current system performance metrics.
    
    Returns:
        SystemMetrics: Current system metrics
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    gpu_memory_mb = 0.0
    gpu_utilization = 0.0
    
    if torch.cuda.is_available():
        try:
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            # Note: GPU utilization requires nvidia-ml-py package for accurate readings
            # For now, we'll use a placeholder
            gpu_utilization = 0.0
        except Exception:
            pass
    
    return SystemMetrics(
        timestamp=time.time(),
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        gpu_memory_mb=gpu_memory_mb,
        gpu_utilization=gpu_utilization,
        disk_usage_percent=disk.percent
    )

@contextmanager
def timer(description: str = "Operation"):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"{description} completed in {elapsed_time:.2f}s")

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path (str): Output file path
        indent (int): JSON indentation
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Data saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise RuntimeError(f"Failed to save data to {file_path}") from e

def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """
    Validate and return Path object.
    
    Args:
        file_path (str): File path to validate
        must_exist (bool): Whether file must exist
        
    Returns:
        Path: Validated path object
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    return path

def cleanup_old_checkpoints(output_dir: str, max_checkpoints: int = 3) -> None:
    """
    Clean up old model checkpoints, keeping only the most recent ones.
    
    Args:
        output_dir (str): Directory containing checkpoints
        max_checkpoints (int): Maximum number of checkpoints to keep
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return
        
        # Find all checkpoint directories
        checkpoint_dirs = [d for d in output_path.iterdir() 
                          if d.is_dir() and d.name.startswith('checkpoint-')]
        
        if len(checkpoint_dirs) <= max_checkpoints:
            return
        
        # Sort by modification time (newest first)
        checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_dirs[max_checkpoints:]:
            try:
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
                
    except Exception as e:
        logger.error(f"Error during checkpoint cleanup: {e}")

def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB" 
