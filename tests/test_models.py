import unittest
from unittest.mock import patch, MagicMock
import torch
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DUMMY_MODEL_NAME = 'dummy-model'
PAD_TOKEN = '[PAD]'
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0

# Adjust the path to import AdvancedModel correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import AdvancedModel


class TestAdvancedModel(unittest.TestCase):
    """Unit tests for the AdvancedModel class."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        logging.debug("Setting up class-level fixtures")

    @classmethod
    def tearDownClass(cls):
        """Tear down class-level fixtures."""
        logging.debug("Tearing down class-level fixtures")

    def setUp(self):
        """Set up test environment."""
        logging.debug("Setting up test environment")
        # Common device for all tests
        self.device = torch.device('cpu')

    def tearDown(self):
        """Clean up after test."""
        logging.debug("Tearing down test environment")

    @patch('main.LlamaForCausalLM')
    @patch('main.LlamaTokenizer')
    def test_initialization_with_pad_token(self, mock_tokenizer_class, mock_model_class):
        """Test AdvancedModel initialization when pad_token is already set."""
        logging.debug("Testing initialization with pad token")

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = PAD_TOKEN
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        logging.debug(f"Mock tokenizer pad token: {mock_tokenizer_instance.pad_token}")

        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        logging.debug(f"Creating AdvancedModel with device: {self.device}")
        model = AdvancedModel(DUMMY_MODEL_NAME, self.device)

        logging.debug("Asserting method calls and attributes")
        mock_tokenizer_class.from_pretrained.assert_called_once_with(DUMMY_MODEL_NAME)
        mock_model_class.from_pretrained.assert_called_once_with(DUMMY_MODEL_NAME)
        self.assertEqual(model.device, self.device)
        mock_tokenizer_instance.add_special_tokens.assert_not_called()
        mock_model_instance.resize_token_embeddings.assert_not_called()
        logging.debug("Initialization with pad token assertions passed")

    @patch('main.LlamaForCausalLM')
    @patch('main.LlamaTokenizer')
    def test_initialization_without_pad_token(self, mock_tokenizer_class, mock_model_class):
        """Test AdvancedModel initialization when pad_token is not set."""
        logging.debug("Testing initialization without pad token")

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        logging.debug(f"Mock tokenizer pad token: {mock_tokenizer_instance.pad_token}")

        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        logging.debug(f"Creating AdvancedModel with device: {self.device}")
        model = AdvancedModel(DUMMY_MODEL_NAME, self.device)

        logging.debug("Asserting method calls and attributes")
        mock_tokenizer_class.from_pretrained.assert_called_once_with(DUMMY_MODEL_NAME)
        mock_model_class.from_pretrained.assert_called_once_with(DUMMY_MODEL_NAME)
        self.assertEqual(model.device, self.device)
        mock_tokenizer_instance.add_special_tokens.assert_called_once_with({'pad_token': PAD_TOKEN})
        mock_model_instance.resize_token_embeddings.assert_called_once_with(len(mock_tokenizer_instance))
        logging.debug("Initialization without pad token assertions passed")

    @patch('main.LlamaForCausalLM')
    @patch('main.LlamaTokenizer')
    def test_forward_pass(self, mock_tokenizer_class, mock_model_class):
        """Test the forward pass of AdvancedModel."""
        logging.debug("Testing forward pass")

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance

        # Create a mock output for the forward pass
        mock_logits = torch.tensor([[0.1, 0.2, 0.3]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits

        # Set up the mock model's forward pass
        mock_model_instance.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model_instance

        logging.debug(f"Creating AdvancedModel with device: {self.device}")
        model = AdvancedModel(DUMMY_MODEL_NAME, self.device)

        # Prepare inputs
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        logging.debug(f"Input IDs: {input_ids}, Attention Mask: {attention_mask}")

        logging.debug("Performing forward pass")
        output = model(input_ids, attention_mask)

        logging.debug("Asserting method calls and output")
        mock_model_instance.assert_called_once_with(input_ids=input_ids, attention_mask=attention_mask)
        torch.testing.assert_close(output, mock_logits)
        logging.debug("Forward pass assertions passed")

    @patch('main.LlamaForCausalLM')
    @patch('main.LlamaTokenizer')
    def test_generate_text(self, mock_tokenizer_class, mock_model_class):
        """Test the generate_text method of AdvancedModel."""
        logging.debug("Testing generate_text method")

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = PAD_TOKEN_ID
        mock_tokenizer_instance.eos_token_id = EOS_TOKEN_ID
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        logging.debug(f"Mock tokenizer pad_token_id: {mock_tokenizer_instance.pad_token_id}, eos_token_id: {mock_tokenizer_instance.eos_token_id}")

        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance

        # Create a mock generated output
        generated_ids = torch.tensor([[4, 5, 6]])
        mock_model_instance.generate.return_value = generated_ids
        logging.debug(f"Mock generated IDs: {generated_ids}")

        mock_model_class.from_pretrained.return_value = mock_model_instance

        logging.debug(f"Creating AdvancedModel with device: {self.device}")
        model = AdvancedModel(DUMMY_MODEL_NAME, self.device)

        # Prepare inputs
        inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        logging.debug(f"Input: {inputs}")

        logging.debug("Calling generate_text method")
        output = model.generate_text(inputs, max_length=10, temperature=0.5, num_return_sequences=2)

        logging.debug("Asserting generate method call and output")
        mock_model_instance.generate.assert_called_once_with(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=10,
            temperature=0.5,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=2,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID
        )
        torch.testing.assert_close(output, generated_ids)
        logging.debug("Generate text assertions passed")


if __name__ == '__main__':
    unittest.main()