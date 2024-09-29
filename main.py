import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
)
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import math
import re
from sympy import simplify, SympifyError, Eq
from sympy.parsing.sympy_parser import parse_expr
import threading
import time
from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
import secrets

# Set random seeds for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    secrets.SystemRandom().seed(seed)

set_seed(42)

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    # Hyperparameters
    beta_1: float = 0.01  # KL divergence coefficient for general training
    beta_2: float = 0.1   # KL divergence coefficient for Stage I (set to 10 * beta_1)
    alpha: float = 5.0    # Reward shaping multiplier (significantly larger than 1.0)
    learning_rate: float = 1e-5
    batch_size: int = 1  # Reduced for LLaMA due to memory constraints
    max_seq_len: int = 1024
    num_epochs_stage_one: int = 1  # Number of epochs for Stage I
    num_epochs_stage_two: int = 1  # Number of epochs for Stage II
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = 42
    task: str = 'MATH'
    model_variant: str = 'decapoda-research/llama-7b-hf'  # Specify the LLaMA model
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
    mixed_precision: bool = False  # Set to True if you want to use mixed precision
    save_total_limit: int = 2

# Data Processing

class MATHDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train', max_samples: int = None):
        self.data = self.load_math_data(data_path, split, max_samples)

    @staticmethod
    def load_math_data(data_path: str, split: str, max_samples: int = None) -> List[Dict[str, Any]]:
        dataset_file = os.path.join(data_path, f'math_{split}.json')
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        if max_samples:
            data = data[:max_samples]
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        problem = self.data[idx]
        return {
            'question': problem['question'],
            'answer': problem['answer']
        }

class MBPPDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train', max_samples: int = None):
        self.data = self.load_mbpp_data(data_path, split, max_samples)

    @staticmethod
    def load_mbpp_data(data_path: str, split: str, max_samples: int = None) -> List[Dict[str, Any]]:
        dataset_file = os.path.join(data_path, f'mbpp_{split}.json')
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        if max_samples:
            data = data[:max_samples]
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        problem = self.data[idx]
        return {
            'text': problem['text'],
            'code': problem['code'],
            'test_list': problem['test_list']
        }

class HumanEvalDataset(Dataset):
    def __init__(self, data_path: str, max_samples: int = None):
        self.data = self.load_humaneval_data(data_path, max_samples)

    @staticmethod
    def load_humaneval_data(data_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
        dataset_file = os.path.join(data_path, 'HumanEval.jsonl')
        data = []
        with open(dataset_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        if max_samples:
            data = data[:max_samples]
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        problem = self.data[idx]
        return {
            'prompt': problem['prompt'],
            'canonical_solution': problem['canonical_solution'],
            'test': problem['test']
        }

# Model Architecture

class AdvancedModel(nn.Module):
    def __init__(self, model_name: str, device: torch.device):
        super(AdvancedModel, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        # LLaMA tokenizer may not have special tokens by default, add them if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model_name = model_name
        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.logits

    def generate(self, inputs: Dict[str, torch.Tensor], max_length: int = 512, temperature: float = 0.7,
                 num_return_sequences: int = 1) -> torch.Tensor:
        generated = self.model.generate(
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
        return generated

# SCoRe Implementation

class SCoReTrainer:
    def __init__(self, model: AdvancedModel, ref_model: AdvancedModel, optimizer: AdamW, scheduler,
                 data_loader: DataLoader, val_loader: DataLoader, config: Config):
        self.model = model
        self.tokenizer = model.tokenizer
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.config = config
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.global_step = 0
        self.reward_history = []
        self.edit_distance_ratios = []
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    def compute_kl_divergence(self, logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        ref_probs = nn.functional.softmax(ref_logits, dim=-1)
        kl_div = self.kl_loss_fn(log_probs, ref_probs)
        return kl_div

    def reward_function_math(self, generated_answer: str, correct_answer: str) -> float:
        try:
            gen_expr = parse_expr(generated_answer, evaluate=True)
            corr_expr = parse_expr(correct_answer, evaluate=True)
            equivalence = simplify(gen_expr - corr_expr) == 0
            return 1.0 if equivalence else 0.0
        except (SympifyError, TypeError):
            return 0.0

    def safe_execute_code(self, code_str: str, test_str: str, timeout: int = 5) -> float:
        def target():
            try:
                exec_globals = {}
                exec(code_str, exec_globals)
                exec(test_str, exec_globals)
                self.exec_result = True
            except Exception:
                self.exec_result = False

        self.exec_result = False
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return False
        return self.exec_result

    def reward_function_code(self, generated_code: str, test_code: str) -> float:
        success = self.safe_execute_code(generated_code, test_code)
        return 1.0 if success else 0.0

    def compute_rewards(self, generated_texts: List[str], correct_answers: List[str], task: str,
                        test_cases: List[str] = None) -> torch.Tensor:
        rewards = []
        if task == 'MATH':
            for gen_ans, corr_ans in zip(generated_texts, correct_answers):
                reward = self.reward_function_math(gen_ans, corr_ans)
                rewards.append(reward)
        elif task == 'CODE':
            for gen_code, test_code in zip(generated_texts, test_cases):
                reward = self.reward_function_code(gen_code, test_code)
                rewards.append(reward)
        return torch.tensor(rewards).to(self.config.device)

    def compute_edit_distance_ratio(self, s1: str, s2: str) -> float:
        seq_matcher = SequenceMatcher(None, s1, s2)
        return seq_matcher.ratio()

    def train(self):
        logger.info("Starting Stage I Training")
        for epoch in range(self.config.num_epochs_stage_one):
            logger.info(f"Stage I Epoch {epoch+1}/{self.config.num_epochs_stage_one}")
            self.stage_one()

        logger.info("Starting Stage II Training")
        for epoch in range(self.config.num_epochs_stage_two):
            logger.info(f"Stage II Epoch {epoch+1}/{self.config.num_epochs_stage_two}")
            self.stage_two()

    def stage_one(self):
        # Stage I: Policy Initialization
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        pbar = tqdm(self.data_loader, desc="Stage I Training", disable=False)
        for step, batch in enumerate(pbar):
            self.global_step += 1
            input_texts, correct_answers, test_cases = self.prepare_batch(batch)

            input_encodings = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len
            ).to(self.config.device)

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(input_encodings['input_ids'], input_encodings['attention_mask'])
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_encodings['input_ids'], input_encodings['attention_mask'])
                kl_loss = self.compute_kl_divergence(outputs, ref_outputs)

            # Generate first attempt
            generated_ids = self.model.generate(
                input_encodings,
                max_length=self.config.max_seq_len,
                temperature=0.7
            )

            generated_answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Compute rewards
            rewards = self.compute_rewards(generated_answers, correct_answers, task=self.config.task, test_cases=test_cases)

            # Compute loss
            loss = -torch.mean(rewards) + self.config.beta_2 * kl_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_reward += rewards.mean().item()
            self.reward_history.append(rewards.mean().item())

            if self.global_step % self.config.logging_steps == 0:
                logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}, Reward: {rewards.mean().item():.4f}")

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Reward': f"{rewards.mean().item():.4f}"})

        avg_loss = total_loss / len(self.data_loader)
        avg_reward = total_reward / len(self.data_loader)
        logger.info(f"Stage I Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}")

    def stage_two(self):
        # Stage II: Multi-Turn RL
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        pbar = tqdm(self.data_loader, desc="Stage II Training", disable=False)
        for step, batch in enumerate(pbar):
            self.global_step += 1
            input_texts, correct_answers, test_cases = self.prepare_batch(batch)

            input_encodings = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len
            ).to(self.config.device)

            # First attempt
            first_attempt_ids = self.model.generate(
                input_encodings,
                max_length=self.config.max_seq_len,
                temperature=0.7
            )
            first_attempt_texts = self.tokenizer.batch_decode(first_attempt_ids, skip_special_tokens=True)

            # Prepare inputs for second attempt
            second_input_texts = []
            for inp, attempt in zip(input_texts, first_attempt_texts):
                prompt = f"{inp}\nPrevious Attempt:\n{attempt}\nInstructions: Please correct the above attempt."
                second_input_texts.append(prompt)

            second_input_encodings = self.tokenizer(
                second_input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len
            ).to(self.config.device)

            # Second attempt
            second_attempt_ids = self.model.generate(
                second_input_encodings,
                max_length=self.config.max_seq_len,
                temperature=0.7
            )
            second_attempt_texts = self.tokenizer.batch_decode(second_attempt_ids, skip_special_tokens=True)

            # Compute rewards
            rewards_first = self.compute_rewards(first_attempt_texts, correct_answers, task=self.config.task, test_cases=test_cases)
            rewards_second = self.compute_rewards(second_attempt_texts, correct_answers, task=self.config.task, test_cases=test_cases)

            # Reward shaping
            bonuses = self.config.alpha * (rewards_second - rewards_first)
            total_rewards = rewards_first + rewards_second + bonuses

            # Compute KL divergence
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(second_input_encodings['input_ids'], second_input_encodings['attention_mask'])
                with torch.no_grad():
                    ref_outputs = self.ref_model(second_input_encodings['input_ids'], second_input_encodings['attention_mask'])
                kl_loss = self.compute_kl_divergence(outputs, ref_outputs)

                # Compute loss
                loss = -torch.mean(total_rewards) + self.config.beta_1 * kl_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_reward += total_rewards.mean().item()
            self.reward_history.append(total_rewards.mean().item())

            if self.global_step % self.config.logging_steps == 0:
                logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}, Total Reward: {total_rewards.mean().item():.4f}")

            # Compute edit distance ratios for visualization
            for fa, sa in zip(first_attempt_texts, second_attempt_texts):
                edit_distance = self.compute_edit_distance_ratio(fa, sa)
                self.edit_distance_ratios.append(edit_distance)

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Total Reward': f"{total_rewards.mean().item():.4f}"})

        avg_loss = total_loss / len(self.data_loader)
        avg_reward = total_reward / len(self.data_loader)
        logger.info(f"Stage II Average Loss: {avg_loss:.4f}, Average Total Reward: {avg_reward:.4f}")

    def prepare_batch(self, batch):
        if self.config.task == 'MATH':
            input_texts = [item['question'] for item in batch]
            correct_answers = [item['answer'] for item in batch]
            test_cases = None
        elif self.config.task == 'CODE':
            input_texts = [item.get('text', item.get('prompt')) for item in batch]
            correct_answers = [item.get('code', item.get('canonical_solution')) for item in batch]
            test_cases = [item.get('test_list', item.get('test')) for item in batch]
        else:
            raise ValueError("Invalid task specified.")
        return input_texts, correct_answers, test_cases

    def evaluate(self):
        # Implement evaluation metrics
        self.model.eval()
        total_correct_t1 = 0
        total_correct_t2 = 0
        total_samples = 0
        delta_i_to_c = 0  # Incorrect to correct
        delta_c_to_i = 0  # Correct to incorrect
        pbar = tqdm(self.val_loader, desc="Evaluation", disable=False)
        for batch in pbar:
            input_texts, correct_answers, test_cases = self.prepare_batch(batch)

            input_encodings = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len
            ).to(self.config.device)

            # First attempt
            with torch.no_grad():
                first_attempt_ids = self.model.generate(
                    input_encodings,
                    max_length=self.config.max_seq_len,
                    temperature=0.0  # Greedy decoding
                )
            first_attempt_texts = self.tokenizer.batch_decode(first_attempt_ids, skip_special_tokens=True)

            # Prepare inputs for second attempt
            second_input_texts = []
            for inp, attempt in zip(input_texts, first_attempt_texts):
                prompt = f"{inp}\nPrevious Attempt:\n{attempt}\nInstructions: Please correct the above attempt."
                second_input_texts.append(prompt)

            second_input_encodings = self.tokenizer(
                second_input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len
            ).to(self.config.device)

            # Second attempt
            with torch.no_grad():
                second_attempt_ids = self.model.generate(
                    second_input_encodings,
                    max_length=self.config.max_seq_len,
                    temperature=0.0  # Greedy decoding
                )
            second_attempt_texts = self.tokenizer.batch_decode(second_attempt_ids, skip_special_tokens=True)

            # Compute rewards
            rewards_first = self.compute_rewards(first_attempt_texts, correct_answers, task=self.config.task, test_cases=test_cases)
            rewards_second = self.compute_rewards(second_attempt_texts, correct_answers, task=self.config.task, test_cases=test_cases)

            # Update metrics
            for r1, r2 in zip(rewards_first, rewards_second):
                total_correct_t1 += r1.item()
                total_correct_t2 += r2.item()
                if r1 == 0 and r2 == 1:
                    delta_i_to_c += 1
                elif r1 == 1 and r2 == 0:
                    delta_c_to_i += 1
                total_samples += 1

        accuracy_t1 = total_correct_t1 / total_samples
        accuracy_t2 = total_correct_t2 / total_samples
        delta = accuracy_t2 - accuracy_t1
        delta_i_to_c_frac = delta_i_to_c / total_samples
        delta_c_to_i_frac = delta_c_to_i / total_samples

        logger.info(f"Accuracy@t1: {accuracy_t1:.4f}")
        logger.info(f"Accuracy@t2: {accuracy_t2:.4f}")
        logger.info(f"Δ(t1,t2): {delta:.4f}")
        logger.info(f"Δ_i→c(t1,t2): {delta_i_to_c_frac:.4f}")
        logger.info(f"Δ_c→i(t1,t2): {delta_c_to_i_frac:.4f}")

        # Visualization
        self.plot_reward_history()
        self.plot_edit_distance_ratios()

    def plot_reward_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.reward_history)
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.title('Training Reward Over Time')
        plt.savefig(os.path.join(self.config.output_dir, 'training_reward.png'))
        plt.close()

    def plot_edit_distance_ratios(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.edit_distance_ratios, bins=50)
        plt.xlabel('Edit Distance Ratio')
        plt.ylabel('Frequency')
        plt.title('Edit Distance Ratios between Attempts')
        plt.savefig(os.path.join(self.config.output_dir, 'edit_distance_ratios.png'))
        plt.close()

# Supervised Fine-Tuning Dataset

class SFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_seq_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        input_text = item['input']
        output_text = item['output']
        encoding = self.tokenizer(
            input_text,
            text_target=output_text,
            truncation=True,
            max_length=self.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['labels'].squeeze()
        }

# Main Script

def main():
    parser = argparse.ArgumentParser(description="Advanced SCoRe System with Enhanced Features")
    parser.add_argument('--task', type=str, default='MATH', choices=['MATH', 'CODE'], help='Task to perform')
    parser.add_argument('--model_variant', type=str, default='decapoda-research/llama-7b-hf', help='Model variant to use')
    parser.add_argument('--ablation', type=str, default='none', help='Ablation study to perform')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to datasets')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.task = args.task
    config.model_variant = args.model_variant
    config.ablation = args.ablation
    config.data_path = args.data_path
    config.output_dir = args.output_dir
    config.mixed_precision = args.mixed_precision

    set_seed(config.seed)

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Data Loaders
    if config.task == 'MATH':
        train_dataset = MATHDataset(data_path=config.data_path, split='train', max_samples=1000)  # Reduced for example
        val_dataset = MATHDataset(data_path=config.data_path, split='test', max_samples=100)     # Reduced for example
    elif config.task == 'CODE':
        train_dataset = MBPPDataset(data_path=config.data_path, split='train')
        val_dataset = HumanEvalDataset(data_path=config.data_path)
    else:
        raise ValueError("Invalid task specified.")

    def collate_fn(batch):
        return batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    # Model Initialization
    model_name = config.model_variant
    model = AdvancedModel(model_name, device=config.device)
    ref_model = AdvancedModel(model_name, device=config.device)  # Reference model for KL divergence penalty
    ref_model.model.eval()

    # Optimizer and Scheduler
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=total_steps)

    # Trainer
    trainer = SCoReTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Training and Evaluation
    trainer.train()
    trainer.evaluate()

    # Save the model
    model_output_path = os.path.join(config.output_dir, 'score_model.bin')
    torch.save(model.model.state_dict(), model_output_path)
    logger.info(f"Model saved to {model_output_path}")

if __name__ == '__main__':
    main()
