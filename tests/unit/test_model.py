import unittest
import torch
from src.score_model.model import AdvancedModel

class TestAdvancedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model_name = 'decapoda-research/llama-7b-hf'

    def test_model_initialization(self):
        try:
            model = AdvancedModel(self.model_name, self.device)
            self.assertIsNotNone(model.tokenizer)
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer.pad_token)
        except Exception as e:
            self.fail(f"Model initialization failed: {e}")

    def test_forward_pass(self):
        model = AdvancedModel(self.model_name, self.device)
        test_input = "Hello, world!"
        encodings = model.tokenizer(
            test_input,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        try:
            output = model(encodings['input_ids'], encodings['attention_mask'])
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(len(output.shape), 3)  # [batch, seq_len, vocab_size]
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")

    def test_text_generation(self):
        model = AdvancedModel(self.model_name, self.device)
        test_input = "Solve the equation: 2x + 3 = 7"
        encodings = model.tokenizer(
            test_input,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        try:
            output_ids = model.generate_text(encodings)
            output_text = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            self.assertIsInstance(output_text, list)
            self.assertTrue(len(output_text) > 0)
        except Exception as e:
            self.fail(f"Text generation failed: {e}")

if __name__ == '__main__':
    unittest.main() 