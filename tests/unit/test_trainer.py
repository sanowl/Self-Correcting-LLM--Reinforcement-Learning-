import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.score_model.trainer import SCoReTrainer
from src.score_model.model import AdvancedModel
from src.score_model.config import Config
from src.score_model.dataset import BaseDataset

class TestSCoReTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(
            task='MATH',
            model_variant='decapoda-research/llama-7b-hf',
            batch_size=2,
            max_seq_len=128,
            num_epochs_stage_one=1,
            num_epochs_stage_two=1
        )
        cls.device = cls.config.device
        
        # Create dummy data
        cls.dummy_data = [
            {'question': 'What is 2+2?', 'answer': '4'},
            {'question': 'What is 3+3?', 'answer': '6'}
        ]
        cls.dataset = BaseDataset(cls.dummy_data)
        cls.dataloader = DataLoader(cls.dataset, batch_size=cls.config.batch_size)

    def setUp(self):
        self.model = AdvancedModel(self.config.model_variant, self.device)
        self.ref_model = AdvancedModel(self.config.model_variant, self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)

    def test_trainer_initialization(self):
        try:
            trainer = SCoReTrainer(
                model=self.model,
                ref_model=self.ref_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_loader=self.dataloader,
                val_loader=self.dataloader,
                config=self.config
            )
            self.assertIsNotNone(trainer)
            self.assertEqual(trainer.global_step, 0)
            self.assertEqual(len(trainer.reward_history), 0)
        except Exception as e:
            self.fail(f"Trainer initialization failed: {e}")

    def test_kl_divergence(self):
        trainer = SCoReTrainer(
            model=self.model,
            ref_model=self.ref_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.dataloader,
            val_loader=self.dataloader,
            config=self.config
        )
        
        # Create dummy logits
        logits1 = torch.randn(2, 10, 100)  # [batch, seq_len, vocab_size]
        logits2 = torch.randn(2, 10, 100)
        
        try:
            kl_div = trainer.compute_kl_divergence(logits1, logits2)
            self.assertIsInstance(kl_div, torch.Tensor)
            self.assertEqual(kl_div.dim(), 0)  # Should be a scalar
        except Exception as e:
            self.fail(f"KL divergence computation failed: {e}")

    def test_reward_computation(self):
        trainer = SCoReTrainer(
            model=self.model,
            ref_model=self.ref_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.dataloader,
            val_loader=self.dataloader,
            config=self.config
        )
        
        generated = ["4", "6"]
        correct = ["4", "6"]
        
        try:
            rewards = trainer.compute_rewards(generated, correct, None)
            self.assertIn('rewards', rewards)
            self.assertIn('bleu', rewards)
            self.assertIn('rouge', rewards)
            self.assertIsInstance(rewards['rewards'], torch.Tensor)
        except Exception as e:
            self.fail(f"Reward computation failed: {e}")

if __name__ == '__main__':
    unittest.main() 