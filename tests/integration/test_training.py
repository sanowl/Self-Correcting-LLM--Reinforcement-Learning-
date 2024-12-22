import unittest
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.score_model.config import Config
from src.score_model.model import AdvancedModel
from src.score_model.dataset import BaseDataset
from src.score_model.trainer import SCoReTrainer
from src.score_model.utils import set_seed

class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up configuration
        cls.config = Config(
            task='MATH',
            model_variant='decapoda-research/llama-7b-hf',
            batch_size=2,
            max_seq_len=128,
            num_epochs_stage_one=1,
            num_epochs_stage_two=1,
            output_dir='./test_outputs'
        )
        
        # Create test output directory
        os.makedirs(cls.config.output_dir, exist_ok=True)
        
        # Set seed for reproducibility
        set_seed(cls.config.seed)
        
        # Create dummy data
        cls.train_data = [
            {'question': 'What is 2+2?', 'answer': '4'},
            {'question': 'What is 3+3?', 'answer': '6'},
            {'question': 'What is 4+4?', 'answer': '8'},
            {'question': 'What is 5+5?', 'answer': '10'}
        ]
        cls.val_data = [
            {'question': 'What is 6+6?', 'answer': '12'},
            {'question': 'What is 7+7?', 'answer': '14'}
        ]

    def setUp(self):
        # Create datasets and dataloaders
        self.train_dataset = BaseDataset(self.train_data)
        self.val_dataset = BaseDataset(self.val_data)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Initialize models
        self.model = AdvancedModel(self.config.model_variant, self.config.device)
        self.ref_model = AdvancedModel(self.config.model_variant, self.config.device)
        self.ref_model.model.eval()
        for param in self.ref_model.model.parameters():
            param.requires_grad = False

        # Setup optimizer and scheduler
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        total_steps = len(self.train_loader) * (self.config.num_epochs_stage_one + self.config.num_epochs_stage_two)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

    def test_full_training_cycle(self):
        try:
            trainer = SCoReTrainer(
                model=self.model,
                ref_model=self.ref_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config=self.config
            )

            # Run training
            trainer.train()
            self.assertTrue(len(trainer.reward_history) > 0)

            # Run evaluation
            trainer.evaluate()
            self.assertTrue(len(trainer.edit_distance_ratios) > 0)

            # Check if plots were created
            self.assertTrue(os.path.exists(os.path.join(self.config.output_dir, 'training_reward.png')))
            self.assertTrue(os.path.exists(os.path.join(self.config.output_dir, 'edit_distance_ratios.png')))

            # Save and load model
            model_path = os.path.join(self.config.output_dir, 'test_model.bin')
            torch.save(self.model.model.state_dict(), model_path)
            self.assertTrue(os.path.exists(model_path))

        except Exception as e:
            self.fail(f"Training cycle failed: {e}")

    def tearDown(self):
        # Clean up test outputs
        import shutil
        if os.path.exists(self.config.output_dir):
            shutil.rmtree(self.config.output_dir)

if __name__ == '__main__':
    unittest.main() 