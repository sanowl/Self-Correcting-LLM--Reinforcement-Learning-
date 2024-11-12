import os
import random
import json
import threading
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing_extensions import TypedDict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import radon.complexity as radon_complexity
from sympy import simplify, SympifyError
from sympy.parsing.sympy_parser import parse_expr
import ast

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


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Seed set to {seed}.")
    except Exception as e:
        logger.error(f"Error setting seed: {e}")
        raise RuntimeError("Failed to set seed.") from e


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


class BaseDataset(Dataset):
    """
    Base dataset class for loading data.
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            return self.data[idx]
        except IndexError as e:
            logger.error(f"Index {idx} out of range for dataset of size {len(self.data)}.")
            raise IndexError("Dataset index out of range.") from e
        except Exception as e:
            logger.error(f"Error retrieving item at index {idx}: {e}")
            raise


def load_json(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load data from a JSON or JSONL file.

    Args:
        file_path (str): Path to the JSON or JSONL file.
        max_samples (Optional[int]): Maximum number of samples to load.

    Returns:
        List[Dict[str, Any]]: Loaded data.
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be a non-negative integer or None")

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for idx, line in enumerate(f):
                    if max_samples is not None and idx >= max_samples:
                        break
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
            else:
                file_content = f.read().strip()
                if file_content:
                    loaded_data = json.loads(file_content)
                    if isinstance(loaded_data, list):
                        data = loaded_data[:max_samples] if max_samples else loaded_data
                    else:
                        data = [loaded_data]
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file {file_path}: {e}")
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e
    except Exception as e:
        logger.error(f"Unexpected error while loading JSON from {file_path}: {e}")
        raise RuntimeError(f"Failed to load data from {file_path}") from e

    logger.info(f"Loaded {len(data)} samples from {file_path}.")
    return data


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
        inputs: Dict[str, torch.Tensor],
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


class RewardsDict(TypedDict):
    """
    TypedDict for rewards and related metrics.
    """
    rewards: torch.Tensor
    bleu: List[float]
    rouge: List[Dict[str, float]]
    cyclomatic: List[float]


class SCoReTrainer:
    """
    Trainer class for the SCoRe system.
    """

    def __init__(
        self,
        model: AdvancedModel,
        ref_model: AdvancedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config
    ):
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.global_step = 0
        self.reward_history: List[float] = []
        self.edit_distance_ratios: List[float] = []
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        if config.task == 'MATH':
            self.rouge = Rouge()
            self.smoothing = SmoothingFunction()

    def compute_kl_divergence(self, logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between model logits and reference logits.

        Args:
            logits (torch.Tensor): Logits from the model.
            ref_logits (torch.Tensor): Logits from the reference model.

        Returns:
            torch.Tensor: KL divergence loss.
        """
        try:
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            ref_probs = nn.functional.softmax(ref_logits, dim=-1)
            kl_div = self.kl_loss_fn(log_probs, ref_probs)
            return kl_div
        except Exception as e:
            logger.error(f"Error computing KL divergence: {e}")
            raise RuntimeError("KL divergence computation failed.") from e

    def reward_function_math(self, generated: str, correct: str) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute rewards for math tasks.

        Args:
            generated (str): Generated answer.
            correct (str): Correct answer.

        Returns:
            Tuple containing reward, BLEU score, and ROUGE scores.
        """
        # Initialize default values
        reward = 0.0
        bleu = 0.0
        rouge = {}

        # Compute mathematical correctness
        try:
            eq = simplify(parse_expr(generated) - parse_expr(correct)) == 0
            reward = 1.0 if eq else 0.0
            logger.debug(f"Math reward: {reward}")
        except (SympifyError, TypeError) as e:
            logger.warning(f"SympifyError or TypeError during math reward computation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in reward_function_math: {e}")

        # Compute BLEU score
        if self.config.compute_bleu:
            try:
                bleu = sentence_bleu(
                    [correct.split()],
                    generated.split(),
                    smoothing_function=self.smoothing.method1
                )
                logger.debug(f"BLEU score: {bleu}")
            except Exception as e:
                logger.error(f"Error computing BLEU score: {e}")

        # Compute ROUGE score
        if self.config.compute_rouge:
            try:
                rouge_scores = self.rouge.get_scores(generated, correct)[0]
                rouge = {'f': rouge_scores.get('f', 0.0)}
                logger.debug(f"ROUGE-F1 score: {rouge['f']}")
            except Exception as e:
                logger.error(f"Error computing ROUGE score: {e}")

        return reward, bleu, rouge

    def safe_execute_code(self, code: str, test: str, timeout: int = 5) -> bool:
        """
        Safely execute generated code with a test case.

        Args:
            code (str): Generated code.
            test (str): Test case code.
            timeout (int): Timeout in seconds.

        Returns:
            bool: Execution success status.
        """
        def target(exec_globals: Dict[str, Any]) -> None:
            try:
                exec(code, exec_globals)
                exec(test, exec_globals)
                exec_globals['exec_success'] = True
            except Exception as e:
                logger.warning(f"Execution error: {e}")
                exec_globals['exec_success'] = False

        exec_globals: Dict[str, Any] = {}
        thread = threading.Thread(target=target, args=(exec_globals,), daemon=True)
        try:
            thread.start()
            thread.join(timeout)
            success = exec_globals.get('exec_success', False)
            if not success and thread.is_alive():
                logger.warning("Code execution timed out.")
                return False
            return success
        except Exception as e:
            logger.error(f"Error during code execution thread: {e}")
            return False

    def compute_cyclomatic_complexity(self, code: str) -> float:
        """
        Compute cyclomatic complexity of the given code.

        Args:
            code (str): Code to analyze.

        Returns:
            float: Average cyclomatic complexity.
        """
        try:
            complexity = radon_complexity.cc_visit(code)
            avg_complexity = np.mean([block.complexity for block in complexity]) if complexity else 0.0
            logger.debug(f"Cyclomatic complexity: {avg_complexity}")
            return avg_complexity
        except SyntaxError as e:
            logger.warning(f"SyntaxError while computing cyclomatic complexity: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error computing cyclomatic complexity: {e}")
            return 0.0

    def reward_function_code(self, code: str, test: str) -> Tuple[float, float]:
        """
        Compute rewards for code tasks.

        Args:
            code (str): Generated code.
            test (str): Test case code.

        Returns:
            Tuple containing reward and cyclomatic complexity.
        """
        success = self.safe_execute_code(code, test)
        cyclomatic = self.compute_cyclomatic_complexity(code) if self.config.compute_cyclomatic_complexity else 0.0
        reward = 1.0 if success else 0.0
        logger.debug(f"Code reward: {reward}, Cyclomatic complexity: {cyclomatic}")
        return reward, cyclomatic

    def compute_rewards(
        self,
        generated: List[str],
        correct: List[str],
        test_cases: Optional[List[str]]
    ) -> RewardsDict:
        """
        Compute rewards for a batch of generated outputs.

        Args:
            generated (List[str]): List of generated outputs.
            correct (List[str]): List of correct answers or code.
            test_cases (Optional[List[str]]): List of test cases for code tasks.

        Returns:
            RewardsDict: Dictionary containing rewards and metrics.
        """
        rewards = []
        bleu = []
        rouge = []
        cyclomatic = []

        for i, gen in enumerate(generated):
            try:
                if self.config.task == 'MATH':
                    r, b, ro = self.reward_function_math(gen, correct[i])
                    rewards.append(r)
                    bleu.append(b)
                    rouge.append(ro)
                elif self.config.task == 'CODE':
                    test = test_cases[i] if test_cases and i < len(test_cases) else ''
                    if test:
                        r, c = self.reward_function_code(gen, test)
                    else:
                        logger.warning(f"Missing test case for CODE task at index {i}. Assigning zero reward.")
                        r, c = 0.0, 0.0
                    rewards.append(r)
                    cyclomatic.append(c)
            except Exception as e:
                logger.error(f"Error computing rewards for index {i}: {e}")
                rewards.append(0.0)
                if self.config.task == 'MATH':
                    bleu.append(0.0)
                    rouge.append({})
                elif self.config.task == 'CODE':
                    cyclomatic.append(0.0)

        rewards_tensor = torch.tensor(rewards, device=self.config.device)
        logger.debug(f"Rewards computed: {rewards}")
        return {
            'rewards': rewards_tensor,
            'bleu': bleu,
            'rouge': rouge,
            'cyclomatic': cyclomatic
        }

    def compute_edit_distance_ratio(self, s1: str, s2: str) -> float:
        """
        Compute the edit distance ratio between two strings.

        Args:
            s1 (str): First string.
            s2 (str): Second string.

        Returns:
            float: Edit distance ratio.
        """
        try:
            ratio = SequenceMatcher(None, s1, s2).ratio()
            logger.debug(f"Edit distance ratio between '{s1}' and '{s2}': {ratio}")
            return ratio
        except Exception as e:
            logger.error(f"Error computing edit distance ratio: {e}")
            return 0.0

    def prepare_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], Optional[List[str]]]:
        """
        Prepare a batch of data for processing.

        Args:
            batch (List[Dict[str, Any]]): Batch of data.

        Returns:
            Tuple containing inputs, correct outputs, and test cases (if applicable).
        """
        try:
            if self.config.task == 'MATH':
                inputs = [item['question'] for item in batch]
                correct = [item['answer'] for item in batch]
                tests = None
            elif self.config.task == 'CODE':
                inputs = [item.get('text', item.get('prompt', '')) for item in batch]
                correct = [item.get('code', item.get('canonical_solution', '')) for item in batch]
                tests = [item.get('test_list', item.get('test', '')) for item in batch]
            else:
                raise ValueError("Invalid task specified.")
            logger.debug(f"Batch prepared with {len(inputs)} samples.")
            return inputs, correct, tests
        except KeyError as e:
            logger.error(f"Missing key in batch data: {e}")
            raise KeyError(f"Missing key in batch data: {e}") from e
        except Exception as e:
            logger.error(f"Error preparing batch: {e}")
            raise RuntimeError("Failed to prepare batch.") from e

    def train(self) -> None:
        """
        Train the model through both training stages.
        """
        try:
            logger.info("Starting training process.")
            for epoch in range(self.config.num_epochs_stage_one):
                logger.info(f"Starting Stage I Training - Epoch {epoch + 1}")
                self.stage_one()
            for epoch in range(self.config.num_epochs_stage_two):
                logger.info(f"Starting Stage II Training - Epoch {epoch + 1}")
                self.stage_two()
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def stage_one(self) -> None:
        """
        Stage I training: Train the model with initial rewards.
        """
        self.model.train()
        total_loss, total_reward = 0.0, 0.0

        for batch in tqdm(self.train_loader, desc="Stage I Training"):
            self.global_step += 1
            try:
                inputs, correct, tests = self.prepare_batch(batch)
                encodings = self.model.tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len
                ).to(self.config.device)
            except Exception as e:
                logger.error(f"Error during batch encoding: {e}")
                continue

            try:
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    logits = self.model(encodings['input_ids'], encodings['attention_mask'])
                    with torch.no_grad():
                        ref_logits = self.ref_model(encodings['input_ids'], encodings['attention_mask'])
                    kl_loss = self.compute_kl_divergence(logits, ref_logits)
                    generated_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len)
                    generated = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    rewards_dict = self.compute_rewards(generated, correct, tests)
                    rewards = rewards_dict['rewards']
                    loss = -rewards.mean() + self.config.beta_2 * kl_loss
            except Exception as e:
                logger.error(f"Error during forward and loss computation: {e}")
                continue

            try:
                self.optimizer.zero_grad()
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
            except Exception as e:
                logger.error(f"Error during backward pass or optimization step: {e}")
                continue

            total_loss += loss.item()
            total_reward += rewards.mean().item()
            self.reward_history.append(rewards.mean().item())

            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}, Loss: {loss.item():.4f}, "
                    f"Reward: {rewards.mean().item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        avg_reward = total_reward / len(self.train_loader)
        logger.info(f"Stage I Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}")

    def stage_two(self) -> None:
        """
        Stage II training: Refine the model with additional attempts and bonuses.
        """
        self.model.train()
        total_loss, total_reward = 0.0, 0.0

        for batch in tqdm(self.train_loader, desc="Stage II Training"):
            self.global_step += 1
            try:
                inputs, correct, tests = self.prepare_batch(batch)
                encodings = self.model.tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len
                ).to(self.config.device)
            except Exception as e:
                logger.error(f"Error during batch encoding: {e}")
                continue

            try:
                generated_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len)
                first_attempt = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                second_inputs = [
                    f"{inp}\nPrevious Attempt:\n{att}\nInstructions: Please correct the above attempt."
                    for inp, att in zip(inputs, first_attempt)
                ]
                second_encodings = self.model.tokenizer(
                    second_inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len
                ).to(self.config.device)
                second_generated_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len)
                second_attempt = self.model.tokenizer.batch_decode(second_generated_ids, skip_special_tokens=True)
                rewards_first = self.compute_rewards(first_attempt, correct, tests)['rewards']
                rewards_second = self.compute_rewards(second_attempt, correct, tests)['rewards']
                bonuses = self.config.alpha * (rewards_second - rewards_first)
                total_rewards = rewards_first + rewards_second + bonuses
            except Exception as e:
                logger.error(f"Error during text generation or reward computation: {e}")
                continue

            try:
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    logits = self.model(second_encodings['input_ids'], second_encodings['attention_mask'])
                    with torch.no_grad():
                        ref_logits = self.ref_model(second_encodings['input_ids'], second_encodings['attention_mask'])
                    kl_loss = self.compute_kl_divergence(logits, ref_logits)
                    loss = -total_rewards.mean() + self.config.beta_1 * kl_loss
            except Exception as e:
                logger.error(f"Error during forward and loss computation: {e}")
                continue

            try:
                self.optimizer.zero_grad()
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
            except Exception as e:
                logger.error(f"Error during backward pass or optimization step: {e}")
                continue

            total_loss += loss.item()
            total_reward += total_rewards.mean().item()
            self.reward_history.append(total_rewards.mean().item())

            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}, Loss: {loss.item():.4f}, "
                    f"Total Reward: {total_rewards.mean().item():.4f}"
                )

            # Compute edit distance ratios
            try:
                for fa, sa in zip(first_attempt, second_attempt):
                    ratio = self.compute_edit_distance_ratio(fa, sa)
                    self.edit_distance_ratios.append(ratio)
            except Exception as e:
                logger.error(f"Error computing edit distance ratios: {e}")

        avg_loss = total_loss / len(self.train_loader)
        avg_reward = total_reward / len(self.train_loader)
        logger.info(f"Stage II Average Loss: {avg_loss:.4f}, Average Total Reward: {avg_reward:.4f}")

    def evaluate(self) -> None:
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_correct_t1, total_correct_t2, total_samples = 0.0, 0.0, 0
        delta_i_to_c, delta_c_to_i = 0, 0
        bleu_scores, rouge_scores, cyclomatic_complexities = [], [], []

        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Evaluation"):
                    try:
                        inputs, correct, tests = self.prepare_batch(batch)
                        encodings = self.model.tokenizer(
                            inputs,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_len
                        ).to(self.config.device)
                    except Exception as e:
                        logger.error(f"Error during batch encoding in evaluation: {e}")
                        continue

                    try:
                        # Generate first attempt
                        first_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len, temperature=0.0)
                        first = self.model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)
                        # Generate second attempt based on first
                        second_inputs = [
                            f"{inp}\nPrevious Attempt:\n{att}\nInstructions: Please correct the above attempt."
                            for inp, att in zip(inputs, first)
                        ]
                        second_encodings = self.model.tokenizer(
                            second_inputs,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_len
                        ).to(self.config.device)
                        second_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len, temperature=0.0)
                        second = self.model.tokenizer.batch_decode(second_ids, skip_special_tokens=True)
                        # Compute rewards
                        rewards_first = self.compute_rewards(first, correct, tests)['rewards']
                        rewards_second = self.compute_rewards(second, correct, tests)['rewards']
                    except Exception as e:
                        logger.error(f"Error during text generation or reward computation in evaluation: {e}")
                        continue

                    for i in range(len(inputs)):
                        try:
                            r1 = rewards_first[i].item()
                            r2 = rewards_second[i].item()
                            total_correct_t1 += r1
                            total_correct_t2 += r2
                            if r1 == 0 and r2 == 1:
                                delta_i_to_c += 1
                            elif r1 == 1 and r2 == 0:
                                delta_c_to_i += 1
                            total_samples += 1

                            if self.config.task == 'MATH':
                                if self.config.compute_bleu:
                                    bleu_first = self.compute_rewards([first[i]], [correct[i]], tests)['bleu'][0]
                                    bleu_second = self.compute_rewards([second[i]], [correct[i]], tests)['bleu'][0]
                                    bleu_scores.extend([bleu_first, bleu_second])
                                if self.config.compute_rouge:
                                    rouge_first = self.compute_rewards([first[i]], [correct[i]], tests)['rouge'][0].get('f', 0.0)
                                    rouge_second = self.compute_rewards([second[i]], [correct[i]], tests)['rouge'][0].get('f', 0.0)
                                    rouge_scores.extend([rouge_first, rouge_second])
                            elif self.config.task == 'CODE':
                                if self.config.compute_cyclomatic_complexity:
                                    cyclomatic = self.compute_rewards([second[i]], [correct[i]], tests)['cyclomatic'][0]
                                    cyclomatic_complexities.append(cyclomatic)

                            # Compute edit distance ratio
                            ratio = self.compute_edit_distance_ratio(first[i], second[i])
                            self.edit_distance_ratios.append(ratio)
                        except Exception as e:
                            logger.error(f"Error during evaluation metrics computation for sample {i}: {e}")

            # Compute final metrics
            accuracy_t1 = total_correct_t1 / total_samples if total_samples > 0 else 0.0
            accuracy_t2 = total_correct_t2 / total_samples if total_samples > 0 else 0.0
            delta = accuracy_t2 - accuracy_t1
            delta_i_to_c_frac = delta_i_to_c / total_samples if total_samples > 0 else 0.0
            delta_c_to_i_frac = delta_c_to_i / total_samples if total_samples > 0 else 0.0

            logger.info(f"Accuracy@t1: {accuracy_t1:.4f}")
            logger.info(f"Accuracy@t2: {accuracy_t2:.4f}")
            logger.info(f"Δ(t1,t2): {delta:.4f}")
            logger.info(f"Δ_i→c(t1,t2): {delta_i_to_c_frac:.4f}")
            logger.info(f"Δ_c→i(t1,t2): {delta_c_to_i_frac:.4f}")

            if self.config.task == 'MATH':
                if self.config.compute_bleu and bleu_scores:
                    avg_bleu = np.mean(bleu_scores)
                    logger.info(f"Average BLEU Score: {avg_bleu:.4f}")
                if self.config.compute_rouge and rouge_scores:
                    avg_rouge = np.mean([score for score in rouge_scores if score is not None])
                    logger.info(f"Average ROUGE-F1 Score: {avg_rouge:.4f}")
            elif self.config.task == 'CODE':
                if self.config.compute_cyclomatic_complexity and cyclomatic_complexities:
                    avg_cyclomatic = np.mean(cyclomatic_complexities)
                    logger.info(f"Average Cyclomatic Complexity: {avg_cyclomatic:.4f}")

            self.plot_reward_history()
            self.plot_edit_distance_ratios()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def plot_reward_history(self) -> None:
        """
        Plot and save the training reward history.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.reward_history, label='Average Reward')
            plt.xlabel('Training Steps')
            plt.ylabel('Average Reward')
            plt.title('Training Reward Over Time')
            plt.legend()
            plt.tight_layout()
            reward_path = os.path.join(self.config.output_dir, 'training_reward.png')
            plt.savefig(reward_path)
            plt.close()
            logger.info(f"Saved reward history plot to {reward_path}.")
        except Exception as e:
            logger.error(f"Error plotting reward history: {e}")

    def plot_edit_distance_ratios(self) -> None:
        """
        Plot and save the histogram of edit distance ratios.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.hist(self.edit_distance_ratios, bins=50, color='skyblue', edgecolor='black')
            plt.xlabel('Edit Distance Ratio')
            plt.ylabel('Frequency')
            plt.title('Edit Distance Ratios between Attempts')
            plt.tight_layout()
            edit_distance_path = os.path.join(self.config.output_dir, 'edit_distance_ratios.png')
            plt.savefig(edit_distance_path)
            plt.close()
            logger.info(f"Saved edit distance ratios plot to {edit_distance_path}.")
        except Exception as e:
            logger.error(f"Error plotting edit distance ratios: {e}")


def main():
    """
    Main function to parse arguments and initiate training and evaluation.
    """
    parser = argparse.ArgumentParser(description="Advanced SCoRe System with Enhanced Features")
    parser.add_argument('--task', type=str, default='MATH', choices=['MATH', 'CODE'], help="Task type: MATH or CODE")
    parser.add_argument('--model_variant', type=str, default='decapoda-research/llama-7b-hf', help="Model variant to use")
    parser.add_argument('--ablation', type=str, default='none', help="Ablation setting")
    parser.add_argument('--data_path', type=str, default='./data', help="Path to the data directory")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save outputs")
    parser.add_argument('--mixed_precision', action='store_true', help="Enable mixed precision training")
    parser.add_argument('--no_bleu', action='store_false', dest='compute_bleu', help="Disable BLEU score computation")
    parser.add_argument('--no_rouge', action='store_false', dest='compute_rouge', help="Disable ROUGE score computation")
    parser.add_argument('--no_cyclomatic', action='store_false', dest='compute_cyclomatic_complexity', help="Disable cyclomatic complexity computation")
    args = parser.parse_args()

    # Initialize configuration
    config = Config(
        task=args.task,
        model_variant=args.model_variant,
        ablation=args.ablation,
        data_path=args.data_path,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        compute_bleu=args.compute_bleu,
        compute_rouge=args.compute_rouge,
        compute_cyclomatic_complexity=args.compute_cyclomatic_complexity
    )

    try:
        config.validate()
    except Exception as e:
        logger.critical(f"Configuration validation failed: {e}")
        return

    try:
        set_seed(config.seed)
    except Exception as e:
        logger.critical(f"Failed to set seed: {e}")
        return

    # Determine data files based on task
    if config.task == 'MATH':
        train_file = os.path.join(config.data_path, 'math_train.json')
        val_file = os.path.join(config.data_path, 'math_test.json')
    elif config.task == 'CODE':
        train_file = os.path.join(config.data_path, 'mbpp_train.json')
        val_file = os.path.join(config.data_path, 'HumanEval.jsonl')
    else:
        logger.critical("Invalid task specified. Choose between 'MATH' and 'CODE'.")
        return

    # Check data file existence
    for file in [train_file, val_file]:
        if not os.path.isfile(file):
            logger.critical(f"Data file {file} does not exist.")
            return

    # Load datasets
    try:
        if config.task == 'MATH':
            train_data = load_json(train_file, 1000)
            val_data = load_json(val_file, 100)
        elif config.task == 'CODE':
            train_data = load_json(train_file, 1000)
            val_data = load_json(val_file, 100)
        train_dataset = BaseDataset(train_data)
        val_dataset = BaseDataset(val_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        logger.info("Datasets loaded successfully.")
    except Exception as e:
        logger.critical(f"Error loading data: {e}")
        return

    # Initialize models
    try:
        model = AdvancedModel(config.model_variant, config.device)
        ref_model = AdvancedModel(config.model_variant, config.device)
        ref_model.model.eval()
        for param in ref_model.model.parameters():
            param.requires_grad = False
        logger.info("Models initialized successfully.")
    except Exception as e:
        logger.critical(f"Error initializing models: {e}")
        return

    # Setup optimizer and scheduler
    try:
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
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
        logger.info("Optimizer and scheduler set up successfully.")
    except Exception as e:
        logger.critical(f"Error setting up optimizer and scheduler: {e}")
        return

    # Initialize trainer
    try:
        trainer = SCoReTrainer(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.critical(f"Error initializing trainer: {e}")
        return

    # Start training and evaluation
    try:
        trainer.train()
        trainer.evaluate()
    except Exception as e:
        logger.critical(f"Error during training/evaluation: {e}")
        return

    # Save the trained model
    try:
        model_save_path = os.path.join(config.output_dir, 'score_model.bin')
        torch.save(model.model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}.")
    except Exception as e:
        logger.critical(f"Error saving the model: {e}")
        return


def load_model(model_path: str, model_variant: str, device: torch.device) -> AdvancedModel:
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model state dict.
        model_variant (str): Model variant identifier.
        device (torch.device): Device to load the model onto.

    Returns:
        AdvancedModel: Loaded model instance.
    """
    try:
        advanced_model = AdvancedModel(model_variant, device)
        advanced_model.model.load_state_dict(torch.load(model_path, map_location=device))
        advanced_model.model.to(device)
        advanced_model.model.eval()
        logger.info(f"Model loaded from {model_path} and moved to {device}.")
        return advanced_model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}") from e
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}") from e


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        raise
