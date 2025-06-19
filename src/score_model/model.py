import torch
import torch.nn as nn
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Dict, Optional, Any, Union
import time
import gc
from contextlib import contextmanager
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelHealth:
    """Model health status tracking."""
    is_healthy: bool = True
    last_check: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

class AdvancedModel(nn.Module):
    """
    Enhanced model wrapper with memory management, health monitoring, and robust error handling.
    """

    def __init__(
        self, 
        model_name: str, 
        device: torch.device,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        enable_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.health = ModelHealth()
        self._lock = threading.Lock()
        
        # Initialize components with retry logic
        self._initialize_tokenizer()
        self._initialize_model()
        self._setup_model_optimizations()
        
        logger.info(f"AdvancedModel initialized successfully for {model_name}")

    def _initialize_tokenizer(self) -> None:
        """Initialize tokenizer with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                logger.info(f"Tokenizer loaded for {self.model_name}")
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to load tokenizer after {self.max_retries} attempts: {e}")
                    self.health.error_count += 1
                    self.health.last_error = f"Tokenizer load failed: {str(e)}"
                    raise RuntimeError(f"Failed to load tokenizer for {self.model_name}") from e
                else:
                    wait_time = 2 ** attempt
                    logger.warning(f"Tokenizer load attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

        # Setup tokenizer padding
        self._setup_tokenizer_padding()

    def _initialize_model(self) -> None:
        """Initialize model with retry logic and memory management."""
        for attempt in range(self.max_retries):
            try:
                # Clear cache before loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map=None  # We'll handle device placement manually
                )
                
                # Move to device
                self.model = self.model.to(self.device)
                logger.info(f"Model loaded and moved to {self.device}")
                break
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to load model after {self.max_retries} attempts: {e}")
                    self.health.error_count += 1
                    self.health.last_error = f"Model load failed: {str(e)}"
                    raise RuntimeError(f"Failed to load model {self.model_name}") from e
                else:
                    wait_time = 2 ** attempt
                    logger.warning(f"Model load attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(wait_time)

    def _setup_tokenizer_padding(self) -> None:
        """Setup tokenizer padding with error handling."""
        try:
            if not self.tokenizer.pad_token:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info("Added pad token and resized token embeddings")
        except Exception as e:
            logger.error(f"Error setting up tokenizer padding: {e}")
            self.health.error_count += 1
            self.health.last_error = f"Tokenizer padding setup failed: {str(e)}"
            raise RuntimeError("Failed to setup tokenizer padding") from e

    def _setup_model_optimizations(self) -> None:
        """Setup model optimizations like gradient checkpointing."""
        try:
            if self.enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

    @contextmanager
    def _memory_management(self):
        """Context manager for memory management during inference."""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with error handling and memory management.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits from the model.
        """
        with self._lock:
            try:
                with self._memory_management():
                    # Validate inputs
                    self._validate_inputs(input_ids, attention_mask)
                    
                    # Ensure inputs are on correct device
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    return outputs.logits
                    
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory during forward pass: {e}")
                self.health.error_count += 1
                self.health.last_error = f"OOM error: {str(e)}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("Out of memory during forward pass") from e
            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                self.health.error_count += 1
                self.health.last_error = f"Forward pass failed: {str(e)}"
                raise RuntimeError("Forward pass failed") from e

    def generate_text(
        self,
        inputs: Dict[str, torch.Tensor],
        max_length: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Enhanced text generation with memory management and error handling.

        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized inputs.
            max_length (int): Maximum length of generated text.
            temperature (float): Sampling temperature.
            num_return_sequences (int): Number of sequences to generate.
            **generation_kwargs: Additional generation parameters.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        with self._lock:
            try:
                with self._memory_management():
                    # Validate inputs
                    if 'input_ids' not in inputs or 'attention_mask' not in inputs:
                        raise ValueError("inputs must contain 'input_ids' and 'attention_mask'")
                    
                    # Ensure inputs are on correct device
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    # Set default generation parameters
                    generation_params = {
                        'max_length': max_length,
                        'temperature': temperature,
                        'do_sample': True,
                        'top_p': 0.95,
                        'num_return_sequences': num_return_sequences,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        **generation_kwargs
                    }
                    
                    # Generate text
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            **generation_params
                        )
                    
                    return generated_ids
                    
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory during text generation: {e}")
                self.health.error_count += 1
                self.health.last_error = f"OOM error during generation: {str(e)}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("Out of memory during text generation") from e
            except Exception as e:
                logger.error(f"Error during text generation: {e}")
                self.health.error_count += 1
                self.health.last_error = f"Text generation failed: {str(e)}"
                raise RuntimeError("Text generation failed") from e

    def _validate_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        """Validate input tensors."""
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}")
        
        if input_ids.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got input_ids with {input_ids.dim()} dimensions")

    def health_check(self) -> ModelHealth:
        """
        Perform comprehensive health check of the model.
        
        Returns:
            ModelHealth: Current health status
        """
        try:
            self.health.last_check = time.time()
            
            # Check memory usage
            if torch.cuda.is_available():
                self.health.gpu_memory_usage_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            
            # Simple forward pass test
            test_input = torch.tensor([[1, 2, 3]], device=self.device)
            test_mask = torch.tensor([[1, 1, 1]], device=self.device)
            
            with torch.no_grad():
                _ = self.model(input_ids=test_input, attention_mask=test_mask)
            
            self.health.is_healthy = True
            logger.debug("Health check passed")
            
        except Exception as e:
            self.health.is_healthy = False
            self.health.error_count += 1
            self.health.last_error = f"Health check failed: {str(e)}"
            logger.error(f"Health check failed: {e}")
        
        return self.health

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dict[str, float]: Memory usage in MB
        """
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats['gpu_allocated'] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            memory_stats['gpu_reserved'] = torch.cuda.memory_reserved(self.device) / 1024 / 1024
            memory_stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        
        return memory_stats

    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memory cache cleared")

    def save_pretrained(self, save_path: str) -> None:
        """
        Save the model in a format compatible with transformers.
        
        Args:
            save_path (str): Path to save the model
        """
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model and tokenizer saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise RuntimeError(f"Failed to save model to {save_path}") from e

    def enable_training_mode(self) -> None:
        """Enable training mode with optimizations."""
        self.model.train()
        if self.enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def enable_evaluation_mode(self) -> None:
        """Enable evaluation mode."""
        self.model.eval()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'parameter_count': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'health': self.health,
            'memory_usage': self.get_memory_usage()
        }
        return info

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            self.clear_cache()
        except:
            pass 