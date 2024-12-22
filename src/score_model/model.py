import torch
import torch.nn as nn
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer

logger = logging.getLogger(__name__)

class AdvancedModel(nn.Module):
    """
    Advanced model wrapper with tokenizer and generation capabilities.
    """

    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            logger.info(f"Tokenizer loaded for {model_name}.")
        except Exception as e:
            logger.error(f"Error loading tokenizer for {model_name}: {e}")
            raise RuntimeError(f"Failed to load tokenizer for {model_name}") from e

        try:
            self.model = LlamaForCausalLM.from_pretrained(model_name).to(device)
            logger.info(f"Model loaded and moved to {device}.")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise RuntimeError(f"Failed to load model {model_name}") from e

        try:
            if not self.tokenizer.pad_token:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info("Added pad token and resized token embeddings.")
        except Exception as e:
            logger.error(f"Error adding pad token or resizing embeddings: {e}")
            raise RuntimeError("Failed to add pad token or resize embeddings.") from e

        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits from the model.
        """
        try:
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise RuntimeError("Forward pass failed.") from e

    def generate_text(
        self,
        inputs: dict[str, torch.Tensor],
        max_length: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized inputs.
            max_length (int): Maximum length of generated text.
            temperature (float): Sampling temperature.
            num_return_sequences (int): Number of sequences to generate.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        try:
            return self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise RuntimeError("Text generation failed.") from e 