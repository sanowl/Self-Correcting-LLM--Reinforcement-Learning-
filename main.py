import os
import random
import json
from typing import Any, Dict, List, Optional, Tuple
import threading
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, get_linear_schedule_with_warmup  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore
import argparse
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
from rouge import Rouge  # type: ignore
import radon.complexity as radon_complexity  # type: ignore
from sympy import simplify, SympifyError  # type: ignore
from sympy.parsing.sympy_parser import parse_expr  # type: ignore
import ast
from typing_extensions import TypedDict
import nltk
from typing import List, Dict, Any, Optional
import json

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
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

class BaseDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def load_json(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be a non-negative integer or None")

    data = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if max_samples is not None and idx >= max_samples:
                    break
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
    else:
        with open(file_path, 'r') as f:
            file_content = f.read().strip()
            if file_content:
                data = json.loads(file_content)
                if not isinstance(data, list):
                    data = [data]  # Wrap non-list data in a list
            # If file is empty, data remains an empty list

    if max_samples is not None:
        data = data[:max_samples]

    return data


class AdvancedModel(nn.Module):
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def generate_text(self, inputs: Dict[str, torch.Tensor], max_length: int = 512, temperature: float = 0.7, num_return_sequences: int = 1) -> torch.Tensor:
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

class RewardsDict(TypedDict):
    rewards: torch.Tensor
    bleu: List[float]
    rouge: List[Dict[str, float]]
    cyclomatic: List[float]

class SCoReTrainer:
    def __init__(self, model: AdvancedModel, ref_model: AdvancedModel, optimizer: torch.optim.Optimizer, scheduler: Any, train_loader: DataLoader, val_loader: DataLoader, config: Config):
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
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        ref_probs = nn.functional.softmax(ref_logits, dim=-1)
        return self.kl_loss_fn(log_probs, ref_probs)

    def reward_function_math(self, generated: str, correct: str) -> Tuple[float, float, Dict[str, float]]:
        try:
            eq = simplify(parse_expr(generated) - parse_expr(correct)) == 0
            reward = 1.0 if eq else 0.0
        except (SympifyError, TypeError):
            reward = 0.0
        bleu = sentence_bleu([correct.split()], generated.split(), smoothing_function=self.smoothing.method1) if self.config.compute_bleu else 0.0
        rouge = self.rouge.get_scores(generated, correct)[0] if self.config.compute_rouge else {}
        return reward, bleu, rouge

    def safe_execute_code(self, code: str, test: str, timeout: int = 5) -> bool:
        def target(exec_globals):
            try:
                exec(code, exec_globals)
                exec(test, exec_globals)
                exec_globals['exec_success'] = True
            except Exception as e:
                exec_globals['exec_success'] = False

        exec_globals = {}
        thread = threading.Thread(target=target, args=(exec_globals,), daemon=True)
        thread.start()
        thread.join(timeout)
        return exec_globals.get('exec_success', False)

    def compute_cyclomatic_complexity(self, code: str) -> float:
        try:
            complexity = radon_complexity.cc_visit(code)
            return np.mean([block.complexity for block in complexity]) if complexity else 0.0
        except SyntaxError:
            return 0.0

    def reward_function_code(self, code: str, test: str) -> Tuple[float, float]:
        success = self.safe_execute_code(code, test)
        cyclomatic = self.compute_cyclomatic_complexity(code) if self.config.compute_cyclomatic_complexity else 0.0
        return (1.0 if success else 0.0), cyclomatic

    def compute_rewards(self, generated: List[str], correct: List[str], test_cases: Optional[List[str]]) -> RewardsDict:
        rewards, bleu, rouge, cyclomatic = [], [], [], []
        for i, gen in enumerate(generated):
            if self.config.task == 'MATH':
                r, b, ro = self.reward_function_math(gen, correct[i])
                rewards.append(r)
                bleu.append(b)
                rouge.append(ro)
            elif self.config.task == 'CODE':
                if test_cases and test_cases[i]:
                    r, c = self.reward_function_code(gen, test_cases[i])
                else:
                    logger.warning(f"Missing test case for CODE task at index {i}. Assigning zero reward.")
                    r, c = 0.0, 0.0
                rewards.append(r)
                cyclomatic.append(c)
        return {'rewards': torch.tensor(rewards, device=self.config.device), 'bleu': bleu, 'rouge': rouge, 'cyclomatic': cyclomatic}

    def compute_edit_distance_ratio(self, s1: str, s2: str) -> float:
        return SequenceMatcher(None, s1, s2).ratio()

    def prepare_batch(self, batch: List[Dict[str, Any]]) -> Tuple[List[str], List[str], Optional[List[str]]]:
        if self.config.task == 'MATH':
            return [item['question'] for item in batch], [item['answer'] for item in batch], None
        elif self.config.task == 'CODE':
            return [item.get('text', item.get('prompt', '')) for item in batch], [item.get('code', item.get('canonical_solution', '')) for item in batch], [item.get('test_list', item.get('test', '')) for item in batch]
        else:
            raise ValueError("Invalid task specified.")

    def train(self):
        for _ in range(self.config.num_epochs_stage_one):
            self.stage_one()
        for _ in range(self.config.num_epochs_stage_two):
            self.stage_two()

    def stage_one(self):
        self.model.train()
        total_loss, total_reward = 0.0, 0.0
        for batch in tqdm(self.train_loader, desc="Stage I Training"):
            self.global_step += 1
            inputs, correct, tests = self.prepare_batch(batch)
            encodings = self.model.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_seq_len).to(self.config.device)
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                logits = self.model(encodings['input_ids'], encodings['attention_mask'])
                with torch.no_grad():
                    ref_logits = self.ref_model(encodings['input_ids'], encodings['attention_mask'])
                kl_loss = self.compute_kl_divergence(logits, ref_logits)
                generated_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len, temperature=0.7)
                generated = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                rewards_dict = self.compute_rewards(generated, correct, tests)
                rewards = rewards_dict['rewards']
                loss = -rewards.mean() + self.config.beta_2 * kl_loss
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
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            self.reward_history.append(rewards.mean().item())
            if self.global_step % self.config.logging_steps == 0:
                logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}, Reward: {rewards.mean().item():.4f}")
        logger.info(f"Stage I Average Loss: {total_loss / len(self.train_loader):.4f}, Average Reward: {total_reward / len(self.train_loader):.4f}")

    def stage_two(self):
        self.model.train()
        total_loss, total_reward = 0.0, 0.0
        for batch in tqdm(self.train_loader, desc="Stage II Training"):
            self.global_step += 1
            inputs, correct, tests = self.prepare_batch(batch)
            encodings = self.model.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_seq_len).to(self.config.device)
            generated_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len, temperature=0.7)
            first_attempt = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            second_inputs = [f"{inp}\nPrevious Attempt:\n{att}\nInstructions: Please correct the above attempt." for inp, att in zip(inputs, first_attempt)]
            second_encodings = self.model.tokenizer(second_inputs, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_seq_len).to(self.config.device)
            second_generated_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len, temperature=0.7)
            second_attempt = self.model.tokenizer.batch_decode(second_generated_ids, skip_special_tokens=True)
            rewards_first = self.compute_rewards(first_attempt, correct, tests)['rewards']
            rewards_second = self.compute_rewards(second_attempt, correct, tests)['rewards']
            bonuses = self.config.alpha * (rewards_second - rewards_first)
            total_rewards = rewards_first + rewards_second + bonuses
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                logits = self.model(second_encodings['input_ids'], second_encodings['attention_mask'])
                with torch.no_grad():
                    ref_logits = self.ref_model(second_encodings['input_ids'], second_encodings['attention_mask'])
                kl_loss = self.compute_kl_divergence(logits, ref_logits)
                loss = -total_rewards.mean() + self.config.beta_1 * kl_loss
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
            total_loss += loss.item()
            total_reward += total_rewards.mean().item()
            self.reward_history.append(total_rewards.mean().item())
            if self.global_step % self.config.logging_steps == 0:
                logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}, Total Reward: {total_rewards.mean().item():.4f}")
            for fa, sa in zip(first_attempt, second_attempt):
                self.edit_distance_ratios.append(self.compute_edit_distance_ratio(fa, sa))
        logger.info(f"Stage II Average Loss: {total_loss / len(self.train_loader):.4f}, Average Total Reward: {total_reward / len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        total_correct_t1, total_correct_t2, total_samples = 0.0, 0.0, 0
        delta_i_to_c, delta_c_to_i = 0, 0
        bleu_scores, rouge_scores, cyclomatic_complexities = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation"):
                inputs, correct, tests = self.prepare_batch(batch)
                encodings = self.model.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_seq_len).to(self.config.device)
                first_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len, temperature=0.0)
                first = self.model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)
                second_inputs = [f"{inp}\nPrevious Attempt:\n{att}\nInstructions: Please correct the above attempt." for inp, att in zip(inputs, first)]
                second_encodings = self.model.tokenizer(second_inputs, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_seq_len).to(self.config.device)
                second_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len, temperature=0.0)
                second = self.model.tokenizer.batch_decode(second_ids, skip_special_tokens=True)
                rewards_first = self.compute_rewards(first, correct, tests)['rewards']
                rewards_second = self.compute_rewards(second, correct, tests)['rewards']
                for i in range(len(inputs)):
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
                        bleu_first = self.compute_rewards([first[i]], [correct[i]], tests)['bleu'][0] if self.config.compute_bleu else 0.0
                        rouge_first = self.compute_rewards([first[i]], [correct[i]], tests)['rouge'][0].get('f', 0.0) if self.config.compute_rouge else 0.0
                        bleu_second = self.compute_rewards([second[i]], [correct[i]], tests)['bleu'][0] if self.config.compute_bleu else 0.0
                        rouge_second = self.compute_rewards([second[i]], [correct[i]], tests)['rouge'][0].get('f', 0.0) if self.config.compute_rouge else 0.0
                        bleu_scores.extend([bleu_first, bleu_second])
                        rouge_scores.extend([rouge_first, rouge_second])
                    elif self.config.task == 'CODE':
                        cyclomatic = self.compute_rewards([second[i]], [correct[i]], tests)['cyclomatic'][0] if self.config.compute_cyclomatic_complexity else 0.0
                        cyclomatic_complexities.append(cyclomatic)
                    self.edit_distance_ratios.append(self.compute_edit_distance_ratio(first[i], second[i]))
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
        if self.config.task == 'MATH':
            avg_bleu = np.mean(bleu_scores) if self.config.compute_bleu else None
            avg_rouge = np.mean([score for score in rouge_scores]) if self.config.compute_rouge else None
            if avg_bleu is not None:
                logger.info(f"Average BLEU Score: {avg_bleu:.4f}")
            if avg_rouge is not None:
                logger.info(f"Average ROUGE-F1 Score: {avg_rouge:.4f}")
        elif self.config.task == 'CODE':
            avg_cyclomatic = np.mean(cyclomatic_complexities) if self.config.compute_cyclomatic_complexity else None
            if avg_cyclomatic is not None:
                logger.info(f"Average Cyclomatic Complexity: {avg_cyclomatic:.4f}")
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

def main():
    parser = argparse.ArgumentParser(description="Advanced SCoRe System with Enhanced Features")
    parser.add_argument('--task', type=str, default='MATH', choices=['MATH', 'CODE'])
    parser.add_argument('--model_variant', type=str, default='decapoda-research/llama-7b-hf')
    parser.add_argument('--ablation', type=str, default='none')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--no_bleu', action='store_false', dest='compute_bleu')
    parser.add_argument('--no_rouge', action='store_false', dest='compute_rouge')
    parser.add_argument('--no_cyclomatic', action='store_false', dest='compute_cyclomatic_complexity')
    args = parser.parse_args()
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
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # Verify data file existence
    if config.task == 'MATH':
        train_file = os.path.join(config.data_path, 'math_train.json')
        val_file = os.path.join(config.data_path, 'math_test.json')
    elif config.task == 'CODE':
        train_file = os.path.join(config.data_path, 'mbpp_train.json')
        val_file = os.path.join(config.data_path, 'HumanEval.jsonl')
    else:
        logger.error("Invalid task specified. Choose between 'MATH' and 'CODE'.")
        return

    for file in [train_file, val_file]:
        if not os.path.isfile(file):
            logger.error(f"Data file {file} does not exist.")
            return

    try:
        if config.task == 'MATH':
            train_data = load_json(train_file, 1000)
            val_data = load_json(val_file, 100)
        elif config.task == 'CODE':
            train_data = load_json(train_file, 1000)
            val_data = load_json(val_file, 100)
        train_dataset = BaseDataset(train_data)
        val_dataset = BaseDataset(val_data)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    try:
        model = AdvancedModel(config.model_variant, config.device)
        ref_model = AdvancedModel(config.model_variant, config.device)
        ref_model.model.eval()
        for param in ref_model.model.parameters():
            param.requires_grad = False
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return

    try:
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        total_steps = len(train_loader) * (config.num_epochs_stage_one + config.num_epochs_stage_two)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps)
    except Exception as e:
        logger.error(f"Error setting up optimizer and scheduler: {e}")
        return

    try:
        trainer = SCoReTrainer(model, ref_model, optimizer, scheduler, train_loader, val_loader, config)
        trainer.train()
        trainer.evaluate()
    except Exception as e:
        logger.error(f"Error during training/evaluation: {e}")
        return

    try:
        torch.save(model.model.state_dict(), os.path.join(config.output_dir, 'score_model.bin'))
        logger.info(f"Model saved to {os.path.join(config.output_dir, 'score_model.bin')}")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        return

# Example loading script
def load_model(model_path: str, model_variant: str, device: torch.device) -> AdvancedModel:
    advanced_model = AdvancedModel(model_variant, device)
    advanced_model.model.load_state_dict(torch.load(model_path, map_location=device))
    advanced_model.model.to(device)
    advanced_model.model.eval()
    return advanced_model

if __name__ == '__main__':
    main()
