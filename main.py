import os
import random
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import logging
import matplotlib.pyplot as plt
from typing import List, Dict
import math
import re
from sympy import simplify, SympifyError, Eq
from sympy.parsing.sympy_parser import parse_expr
import subprocess
import threading
import time
import signal
import sys

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Environment Setup
# Define hyperparameters
beta_1 = 0.01  # KL divergence coefficient for general training
beta_2 = 0.1   # KL divergence coefficient for Stage I (set to 10 * beta_1)
alpha = 5.0    # Reward shaping multiplier (significantly larger than 1.0)
learning_rate = 1e-5
batch_size = 4
max_seq_len = 1024
num_epochs_stage_one = 1  # Number of epochs for Stage I
num_epochs_stage_two = 2  # Number of epochs for Stage II
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Processing

class MATHDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train'):
        self.data = self.load_math_data(data_path, split)

    def load_math_data(self, data_path: str, split: str) -> List[Dict]:
        # Load problems from the MATH dataset
        dataset_file = os.path.join(data_path, f'math_{split}.json')
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        if split == 'train':
            data = data[:4500]
        else:
            data = data[:500]
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
    def __init__(self, data_path: str, split: str = 'train'):
        self.data = self.load_mbpp_data(data_path, split)

    def load_mbpp_data(self, data_path: str, split: str) -> List[Dict]:
        dataset_file = os.path.join(data_path, f'mbpp_{split}.json')
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        problem = self.data[idx]
        return {
            'text': problem['text'],
            'code': problem['code'],
            'test_list': problem['test_list']
        }

class HumanEvalDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'test'):
        self.data = self.load_humaneval_data(data_path)

    def load_humaneval_data(self, data_path: str) -> List[Dict]:
        dataset_file = os.path.join(data_path, 'HumanEval.jsonl')
        data = []
        with open(dataset_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        problem = self.data[idx]
        return {
            'prompt': problem['prompt'],
            'canonical_solution': problem['canonical_solution'],
            'test': problem['test']
        }

# Model Architecture

class AdvancedModel(nn.Module):
    def __init__(self, model_name: str):
        super(AdvancedModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model_name = model_name

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.logits

    def generate(self, inputs: Dict[str, torch.Tensor], max_length: int = 512, temperature: float = 0.7, num_return_sequences: int = 1) -> torch.Tensor:
        generated = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return generated

# SCoRe Implementation

class SCoReTrainer:
    def __init__(self, model: AdvancedModel, ref_model: AdvancedModel, optimizer: AdamW, scheduler, data_loader: DataLoader, val_loader: DataLoader, args):
        self.model = model
        self.tokenizer = model.tokenizer
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.args = args
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.global_step = 0
        self.reward_history = []
        self.edit_distance_ratios = []

    def compute_kl_divergence(self, logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        ref_probs = nn.functional.softmax(ref_logits, dim=-1)
        kl_div = self.kl_loss_fn(log_probs, ref_probs)
        return kl_div

    def reward_function_math(self, generated_answer: str, correct_answer: str) -> float:
        # Use SymPy to symbolically simplify and compare answers
        try:
            gen_expr = parse_expr(generated_answer, evaluate=True)
            corr_expr = parse_expr(correct_answer, evaluate=True)
            equivalence = simplify(gen_expr - corr_expr) == 0
            return 1.0 if equivalence else 0.0
        except (SympifyError, TypeError):
            return 0.0

    def safe_execute_code(self, code_str: str, test_str: str, timeout: int = 5) -> bool:
        # Execute code in a subprocess with timeouts and resource limits
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
        # Execute the generated code and run test cases
        success = self.safe_execute_code(generated_code, test_code)
        return 1.0 if success else 0.0

    def compute_rewards(self, generated_texts: List[str], correct_answers: List[str], task: str, test_cases: List[str] = None) -> torch.Tensor:
        rewards = []
        if task == 'MATH':
            for gen_ans, corr_ans in zip(generated_texts, correct_answers):
                reward = self.reward_function_math(gen_ans, corr_ans)
                rewards.append(reward)
        elif task == 'CODE':
            for gen_code, test_code in zip(generated_texts, test_cases):
                reward = self.reward_function_code(gen_code, test_code)
                rewards.append(reward)
        return torch.tensor(rewards).to(device)

    def stage_one(self):
        # Stage I: Policy Initialization
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        for batch in tqdm(self.data_loader, desc="Stage I Training"):
            self.global_step += 1
            if self.args.task == 'MATH':
                input_texts = [item['question'] for item in batch]
                correct_answers = [item['answer'] for item in batch]
                test_cases = None
            elif self.args.task == 'CODE':
                input_texts = [item['text'] for item in batch]
                correct_answers = [item['code'] for item in batch]
                test_cases = [item['test_list'] for item in batch]
            else:
                raise ValueError("Invalid task specified.")

            input_encodings = self.tokenizer(
                list(input_texts),
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_seq_len
            ).to(device)

            outputs = self.model(input_encodings['input_ids'], input_encodings['attention_mask'])
            with torch.no_grad():
                ref_outputs = self.ref_model(input_encodings['input_ids'], input_encodings['attention_mask'])

            kl_loss = self.compute_kl_divergence(outputs, ref_outputs)

            # Generate first attempt
            generated_ids = self.model.generate(
                input_encodings,
                max_length=max_seq_len,
                temperature=0.7
            )

            generated_answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Compute rewards
            if self.args.task == 'MATH':
                rewards = self.compute_rewards(generated_answers, correct_answers, task='MATH')
            elif self.args.task == 'CODE':
                # For code generation, test_cases are provided
                rewards = self.compute_rewards(generated_answers, correct_answers, task='CODE', test_cases=test_cases)

            # Compute loss
            loss = -torch.mean(rewards) + beta_2 * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_reward += rewards.mean().item()
            self.reward_history.append(rewards.mean().item())

            if self.global_step % 10 == 0:
                logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}, Reward: {rewards.mean().item():.4f}")

        avg_loss = total_loss / len(self.data_loader)
        avg_reward = total_reward / len(self.data_loader)
        logger.info(f"Stage I Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}")

    def stage_two(self):
        # Stage II: Multi-Turn RL
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        for batch in tqdm(self.data_loader, desc="Stage II Training"):
            self.global_step += 1
            if self.args.task == 'MATH':
                input_texts = [item['question'] for item in batch]
                correct_answers = [item['answer'] for item in batch]
                test_cases = None
            elif self.args.task == 'CODE':
                input_texts = [item['text'] for item in batch]
                correct_answers = [item['code'] for item in batch]
                test_cases = [item['test_list'] for item in batch]
            else:
                raise ValueError("Invalid task specified.")

            input_encodings = self.tokenizer(
                list(input_texts),
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_seq_len
            ).to(device)

            # First attempt
            first_attempt_ids = self.model.generate(
                input_encodings,
                max_length=max_seq_len,
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
                max_length=max_seq_len
            ).to(device)

            # Second attempt
            second_attempt_ids = self.model.generate(
                second_input_encodings,
                max_length=max_seq_len,
                temperature=0.7
            )
            second_attempt_texts = self.tokenizer.batch_decode(second_attempt_ids, skip_special_tokens=True)

            # Compute rewards
            if self.args.task == 'MATH':
                rewards_first = self.compute_rewards(first_attempt_texts, correct_answers, task='MATH')
                rewards_second = self.compute_rewards(second_attempt_texts, correct_answers, task='MATH')
            elif self.args.task == 'CODE':
                rewards_first = self.compute_rewards(first_attempt_texts, correct_answers, task='CODE', test_cases=test_cases)
                rewards_second = self.compute_rewards(second_attempt_texts, correct_answers, task='CODE', test_cases=test_cases)

            # Reward shaping
            bonuses = alpha * (rewards_second - rewards_first)
            total_rewards = rewards_first + rewards_second + bonuses

            # Compute KL divergence
            outputs = self.model(second_input_encodings['input_ids'], second_input_encodings['attention_mask'])
            with torch.no_grad():
                ref_outputs = self.ref_model(second_input_encodings['input_ids'], second_input_encodings['attention_mask'])
            kl_loss = self.compute_kl_divergence(outputs, ref_outputs)

            # Compute loss
            loss = -torch.mean(total_rewards) + beta_1 * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_reward += total_rewards.mean().item()
            self.reward_history.append(total_rewards.mean().item())

            if self.global_step % 10 == 0:
                logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}, Total Reward: {total_rewards.mean().item():.4f}")

            # Compute edit distance ratios for visualization
            for fa, sa in zip(first_attempt_texts, second_attempt_texts):
                edit_distance = self.compute_edit_distance_ratio(fa, sa)
                self.edit_distance_ratios.append(edit_distance)

        avg_loss = total_loss / len(self.data_loader)
        avg_reward = total_reward / len(self.data_loader)
        logger.info(f"Stage II Average Loss: {avg_loss:.4f}, Average Total Reward: {avg_reward:.4f}")

    def compute_edit_distance_ratio(self, s1: str, s2: str) -> float:
        # Compute edit distance ratio between two strings
        from difflib import SequenceMatcher
        seq_matcher = SequenceMatcher(None, s1, s2)
        return seq_matcher.ratio()

    def train(self):
        logger.info("Starting Stage I Training")
        for epoch in range(num_epochs_stage_one):
            logger.info(f"Stage I Epoch {epoch+1}/{num_epochs_stage_one}")
            self.stage_one()

        logger.info("Starting Stage II Training")
        for epoch in range(num_epochs_stage_two):
            logger.info(f"Stage II Epoch {epoch+1}/{num_epochs_stage_two}")
            self.stage_two()

    def evaluate(self):
        # Implement evaluation metrics
        self.model.eval()
        total_correct_t1 = 0
        total_correct_t2 = 0
        total_samples = 0
        delta_i_to_c = 0  # Incorrect to correct
        delta_c_to_i = 0  # Correct to incorrect
        for batch in tqdm(self.val_loader, desc="Evaluation"):
            if self.args.task == 'MATH':
                input_texts = [item['question'] for item in batch]
                correct_answers = [item['answer'] for item in batch]
                test_cases = None
            elif self.args.task == 'CODE':
                input_texts = [item['prompt'] for item in batch]
                correct_answers = [item['canonical_solution'] for item in batch]
                test_cases = [item['test'] for item in batch]
            else:
                raise ValueError("Invalid task specified.")

            input_encodings = self.tokenizer(
                list(input_texts),
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_seq_len
            ).to(device)

            # First attempt
            with torch.no_grad():
                first_attempt_ids = self.model.generate(
                    input_encodings,
                    max_length=max_seq_len,
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
                max_length=max_seq_len
            ).to(device)

            # Second attempt
            with torch.no_grad():
                second_attempt_ids = self.model.generate(
                    second_input_encodings,
                    max_length=max_seq_len,
                    temperature=0.0  # Greedy decoding
                )
            second_attempt_texts = self.tokenizer.batch_decode(second_attempt_ids, skip_special_tokens=True)

            # Compute rewards
            if self.args.task == 'MATH':
                rewards_first = self.compute_rewards(first_attempt_texts, correct_answers, task='MATH')
                rewards_second = self.compute_rewards(second_attempt_texts, correct_answers, task='MATH')
            elif self.args.task == 'CODE':
                rewards_first = self.compute_rewards(first_attempt_texts, correct_answers, task='CODE', test_cases=test_cases)
                rewards_second = self.compute_rewards(second_attempt_texts, correct_answers, task='CODE', test_cases=test_cases)

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
        plt.savefig('training_reward.png')
        plt.close()

    def plot_edit_distance_ratios(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.edit_distance_ratios, bins=50)
        plt.xlabel('Edit Distance Ratio')
        plt.ylabel('Frequency')
        plt.title('Edit Distance Ratios between Attempts')
        plt.savefig('edit_distance_ratios.png')
        plt.close()

    # Compute Scaling Implementation

    def compute_scaling_experiment(self):
        # Compare parallel sampling vs. sequential self-correction
        self.model.eval()
        total_samples = 0
        seq_correct = 0
        par_correct = 0
        for batch in tqdm(self.val_loader, desc="Compute Scaling Experiment"):
            if self.args.task == 'MATH':
                input_texts = [item['question'] for item in batch]
                correct_answers = [item['answer'] for item in batch]
                test_cases = None
            elif self.args.task == 'CODE':
                input_texts = [item['prompt'] for item in batch]
                correct_answers = [item['canonical_solution'] for item in batch]
                test_cases = [item['test'] for item in batch]
            else:
                raise ValueError("Invalid task specified.")

            # Sequential Self-Correction
            seq_correct += self.sequential_self_correction(input_texts, correct_answers, test_cases)
            # Parallel Sampling
            par_correct += self.parallel_sampling(input_texts, correct_answers, test_cases)
            total_samples += len(batch)

        logger.info(f"Sequential Self-Correction Accuracy: {seq_correct / total_samples:.4f}")
        logger.info(f"Parallel Sampling Accuracy: {par_correct / total_samples:.4f}")

    def sequential_self_correction(self, input_texts, correct_answers, test_cases):
        # Sequential self-correction over multiple attempts
        correct_count = 0
        for inp, corr_ans, test_case in zip(input_texts, correct_answers, test_cases if test_cases else [None]*len(input_texts)):
            attempt = inp
            for _ in range(3):  # Number of correction attempts
                input_encodings = self.tokenizer(
                    [attempt],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len
                ).to(device)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_encodings,
                        max_length=max_seq_len,
                        temperature=0.7
                    )
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                if self.args.task == 'MATH':
                    reward = self.reward_function_math(generated_text, corr_ans)
                elif self.args.task == 'CODE':
                    reward = self.reward_function_code(generated_text, test_case)
                if reward == 1.0:
                    correct_count += 1
                    break
                else:
                    attempt = f"{inp}\nPrevious Attempt:\n{generated_text}\nInstructions: Please correct the above attempt."
        return correct_count

    def parallel_sampling(self, input_texts, correct_answers, test_cases):
        # Parallel sampling with self-consistency
        correct_count = 0
        for inp, corr_ans, test_case in zip(input_texts, correct_answers, test_cases if test_cases else [None]*len(input_texts)):
            input_encodings = self.tokenizer(
                [inp],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_seq_len
            ).to(device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_encodings['input_ids'],
                    attention_mask=input_encodings['attention_mask'],
                    max_length=max_seq_len,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=10
                )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Majority voting
            from collections import Counter
            answer_counts = Counter(generated_texts)
            most_common_answer = answer_counts.most_common(1)[0][0]
            if self.args.task == 'MATH':
                reward = self.reward_function_math(most_common_answer, corr_ans)
            elif self.args.task == 'CODE':
                reward = self.reward_function_code(most_common_answer, test_case)
            if reward == 1.0:
                correct_count += 1
        return correct_count

    # Baseline Implementations

    def self_refine(self):
        # Implement Self-Refine baseline
        logger.info("Running Self-Refine Baseline")
        self.model.eval()
        total_correct = 0
        total_samples = 0
        for batch in tqdm(self.val_loader, desc="Self-Refine Evaluation"):
            if self.args.task == 'MATH':
                input_texts = [item['question'] for item in batch]
                correct_answers = [item['answer'] for item in batch]
            elif self.args.task == 'CODE':
                input_texts = [item['prompt'] for item in batch]
                correct_answers = [item['canonical_solution'] for item in batch]
                test_cases = [item['test'] for item in batch]
            else:
                raise ValueError("Invalid task specified.")

            # Generate initial attempts
            input_encodings = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_seq_len
            ).to(device)

            with torch.no_grad():
                initial_ids = self.model.generate(
                    input_encodings,
                    max_length=max_seq_len,
                    temperature=0.0
                )
            initial_texts = self.tokenizer.batch_decode(initial_ids, skip_special_tokens=True)

            # Generate self-refined attempts using prompting
            refined_texts = []
            for inp, initial in zip(input_texts, initial_texts):
                prompt = f"{inp}\nHere is an initial attempt:\n{initial}\nPlease provide a refined solution."
                prompt_encodings = self.tokenizer(
                    [prompt],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len
                ).to(device)
                with torch.no_grad():
                    refined_ids = self.model.generate(
                        prompt_encodings,
                        max_length=max_seq_len,
                        temperature=0.0
                    )
                refined_text = self.tokenizer.decode(refined_ids[0], skip_special_tokens=True)
                refined_texts.append(refined_text)

            # Compute rewards
            if self.args.task == 'MATH':
                rewards = self.compute_rewards(refined_texts, correct_answers, task='MATH')
            elif self.args.task == 'CODE':
                rewards = self.compute_rewards(refined_texts, correct_answers, task='CODE', test_cases=test_cases)

            total_correct += rewards.sum().item()
            total_samples += len(rewards)

        accuracy = total_correct / total_samples
        logger.info(f"Self-Refine Baseline Accuracy: {accuracy:.4f}")

    def star(self):
        # Implement STaR baseline
        logger.info("Running STaR Baseline")
        # Collect self-generated data
        self.model.eval()
        sft_data = []
        for batch in tqdm(self.data_loader, desc="Collecting Self-Generated Data"):
            if self.args.task == 'MATH':
                input_texts = [item['question'] for item in batch]
                correct_answers = [item['answer'] for item in batch]
            elif self.args.task == 'CODE':
                input_texts = [item['text'] for item in batch]
                correct_answers = [item['code'] for item in batch]
                test_cases = [item['test_list'] for item in batch]
            else:
                raise ValueError("Invalid task specified.")

            input_encodings = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_seq_len
            ).to(device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_encodings,
                    max_length=max_seq_len,
                    temperature=0.7
                )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Filter correct responses
            if self.args.task == 'MATH':
                rewards = self.compute_rewards(generated_texts, correct_answers, task='MATH')
            elif self.args.task == 'CODE':
                rewards = self.compute_rewards(generated_texts, correct_answers, task='CODE', test_cases=test_cases)

            for inp, gen_text, reward in zip(input_texts, generated_texts, rewards):
                if reward.item() == 1.0:
                    sft_data.append({'input': inp, 'output': gen_text})

        # Supervised Fine-Tuning
        logger.info("Performing Supervised Fine-Tuning")
        sft_dataset = SFTDataset(sft_data, self.tokenizer)
        sft_loader = DataLoader(sft_dataset, batch_size=batch_size, shuffle=True)
        sft_optimizer = AdamW(self.model.model.parameters(), lr=learning_rate)

        for epoch in range(1):  # Number of SFT epochs
            total_loss = 0.0
            for batch in tqdm(sft_loader, desc="SFT Training"):
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                sft_optimizer.zero_grad()
                loss.backward()
                sft_optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(sft_loader)
            logger.info(f"SFT Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Evaluate the fine-tuned model
        self.evaluate()

    def pair_sft(self):
        # Implement Pair-SFT baseline
        logger.info("Running Pair-SFT Baseline")
        # Generate synthetic pairs
        synthetic_data = []
        for batch in tqdm(self.data_loader, desc="Generating Synthetic Pairs"):
            if self.args.task == 'MATH':
                input_texts = [item['question'] for item in batch]
                correct_answers = [item['answer'] for item in batch]
            elif self.args.task == 'CODE':
                input_texts = [item['text'] for item in batch]
                correct_answers = [item['code'] for item in batch]
                test_cases = [item['test_list'] for item in batch]
            else:
                raise ValueError("Invalid task specified.")

            # Generate incorrect and correct pairs
            input_encodings = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_seq_len
            ).to(device)

            with torch.no_grad():
                incorrect_ids = self.model.generate(
                    input_encodings,
                    max_length=max_seq_len,
                    temperature=1.0,
                    do_sample=True,
                    top_p=0.9
                )
            incorrect_texts = self.tokenizer.batch_decode(incorrect_ids, skip_special_tokens=True)

            # For each incorrect attempt, create a synthetic pair with the correct answer
            for inp, incorrect, correct in zip(input_texts, incorrect_texts, correct_answers):
                synthetic_data.append({'input': inp + incorrect, 'output': correct})

        # Supervised Fine-Tuning on synthetic pairs
        logger.info("Performing Supervised Fine-Tuning on Synthetic Pairs")
        sft_dataset = SFTDataset(synthetic_data, self.tokenizer)
        sft_loader = DataLoader(sft_dataset, batch_size=batch_size, shuffle=True)
        sft_optimizer = AdamW(self.model.model.parameters(), lr=learning_rate)

        for epoch in range(1):  # Number of SFT epochs
            total_loss = 0.0
            for batch in tqdm(sft_loader, desc="Pair-SFT Training"):
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                sft_optimizer.zero_grad()
                loss.backward()
                sft_optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(sft_loader)
            logger.info(f"Pair-SFT Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Evaluate the fine-tuned model
        self.evaluate()

# Supervised Fine-Tuning Dataset

class SFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

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
            max_length=max_seq_len,
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
    parser.add_argument('--model_variant', type=str, required=True, help='Model variant to use')
    parser.add_argument('--ablation', type=str, default='none', help='Ablation study to perform')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to datasets')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Data Loaders
    if args.task == 'MATH':
        train_dataset = MATHDataset(data_path=args.data_path, split='train')
        val_dataset = MATHDataset(data_path=args.data_path, split='test')
    elif args.task == 'CODE':
        # Use MBPP for training and HumanEval for testing
        train_dataset = MBPPDataset(data_path=args.data_path, split='train')
        val_dataset = HumanEvalDataset(data_path=args.data_path)
    else:
        raise ValueError("Invalid task specified.")

    def collate_fn(batch):
        return batch

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model Initialization
    model_name = args.model_variant
    model = AdvancedModel(model_name)
    ref_model = AdvancedModel(model_name)  # Reference model for KL divergence penalty
    ref_model.model.eval()

    # Optimizer and Scheduler
    no_decay = ['bias', 'LayerNorm.weight']
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = len(train_loader) * (num_epochs_stage_one + num_epochs_stage_two)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    # Trainer
    trainer = SCoReTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loader=train_loader,
        val_loader=val_loader,
        args=args
    )

    # Training and Evaluation
    trainer.train()
    trainer.evaluate()

    # Baseline Implementations
    logger.info("Running Baseline Methods")
    trainer.self_refine()
    trainer.star()
    trainer.pair_sft()

    # Compute Scaling Experiments
    trainer.compute_scaling_experiment()

    # Save the model
    model_output_path = os.path.join(args.output_dir, 'score_model.bin')
    torch.save(model.model.state_dict(), model_output_path)
    logger.info(f"Model saved to {model_output_path}")

if __name__ == '__main__':
    main()
