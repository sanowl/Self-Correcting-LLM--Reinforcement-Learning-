import os
import torch
import torch.nn as nn
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict
from tqdm import tqdm
from difflib import SequenceMatcher
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import radon.complexity as radon_complexity
from sympy import simplify, SympifyError
from sympy.parsing.sympy_parser import parse_expr
import threading
from torch.utils.data import DataLoader

from .config import Config
from .model import AdvancedModel

logger = logging.getLogger(__name__)

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
        """
        reward = 0.0
        bleu = 0.0
        rouge = {}

        try:
            eq = simplify(parse_expr(generated) - parse_expr(correct)) == 0
            reward = 1.0 if eq else 0.0
            logger.debug(f"Math reward: {reward}")
        except (SympifyError, TypeError) as e:
            logger.warning(f"SympifyError or TypeError during math reward computation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in reward_function_math: {e}")

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