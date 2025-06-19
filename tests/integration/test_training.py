import pytest
import tempfile
import os
import json
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil

from src.score_model.config import Config
from src.score_model.model import AdvancedModel
from src.score_model.dataset import BaseDataset
from src.score_model.trainer import SCoReTrainer
from src.score_model.utils import setup_logging, set_seed, load_json


class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return [
            {
                "question": "What is 2+2?",
                "answer": "4",
                "solution": "2+2=4"
            },
            {
                "question": "What is 3*3?", 
                "answer": "9",
                "solution": "3*3=9"
            }
        ]

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration for testing."""
        return Config(
            task='MATH',
            model_variant='dummy-model',
            data_path=os.path.join(temp_dir, 'data'),
            output_dir=os.path.join(temp_dir, 'outputs'),
            log_dir=os.path.join(temp_dir, 'logs'),
            batch_size=1,
            num_epochs_stage_one=1,
            num_epochs_stage_two=1,
            max_seq_len=128,
            learning_rate=1e-4,
            save_steps=1,
            eval_steps=1,
            logging_steps=1,
            environment='development'
        )

    def setup_test_data(self, config, sample_data):
        """Setup test data files."""
        os.makedirs(config.data_path, exist_ok=True)
        
        train_file = os.path.join(config.data_path, 'math_train.json')
        test_file = os.path.join(config.data_path, 'math_test.json')
        
        with open(train_file, 'w') as f:
            json.dump(sample_data, f)
        
        with open(test_file, 'w') as f:
            json.dump(sample_data[:1], f)

    @patch('src.score_model.model.LlamaForCausalLM')
    @patch('src.score_model.model.LlamaTokenizer')
    def test_complete_training_pipeline(self, mock_tokenizer_class, mock_model_class, 
                                       mock_config, sample_data):
        """Test the complete training pipeline end-to-end."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = '[PAD]'
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.__len__ = MagicMock(return_value=1000)
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_model.named_parameters.return_value = [('layer.weight', torch.randn(10, requires_grad=True))]
        
        # Mock forward pass
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 1000)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        # Setup test data
        self.setup_test_data(mock_config, sample_data)
        
        # Validate config
        mock_config.validate()
        
        # Setup logging
        logger = setup_logging(mock_config)
        
        # Set seed
        set_seed(mock_config.seed)
        
        # Load data
        train_file = os.path.join(mock_config.data_path, 'math_train.json')
        val_file = os.path.join(mock_config.data_path, 'math_test.json')
        
        train_data = load_json(train_file)
        val_data = load_json(val_file)
        
        assert len(train_data) == 2
        assert len(val_data) == 1
        
        # Create datasets
        train_dataset = BaseDataset(train_data)
        val_dataset = BaseDataset(val_data)
        
        assert len(train_dataset) == 2
        assert len(val_dataset) == 1
        
        # Initialize models
        model = AdvancedModel(mock_config.model_variant, mock_config.device)
        ref_model = AdvancedModel(mock_config.model_variant, mock_config.device)
        
        # Health checks
        model_health = model.health_check()
        assert model_health.is_healthy
        
        ref_health = ref_model.health_check()
        assert ref_health.is_healthy
        
        logger.info("Integration test completed successfully")

    @patch('src.score_model.model.LlamaForCausalLM')
    @patch('src.score_model.model.LlamaTokenizer')
    def test_model_health_monitoring(self, mock_tokenizer_class, mock_model_class, mock_config):
        """Test model health monitoring functionality."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = '[PAD]'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        
        # Mock successful forward pass for health check
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 3, 1000)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        mock_config.validate()
        
        # Initialize model
        model = AdvancedModel(mock_config.model_variant, mock_config.device)
        
        # Test health check
        health = model.health_check()
        
        assert health.is_healthy
        assert health.last_check > 0
        assert health.error_count == 0
        assert health.last_error is None

    @patch('src.score_model.model.LlamaForCausalLM')
    @patch('src.score_model.model.LlamaTokenizer')
    def test_model_error_handling(self, mock_tokenizer_class, mock_model_class, mock_config):
        """Test model error handling during health checks."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = '[PAD]'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        
        # Mock failing forward pass for health check
        mock_model.side_effect = RuntimeError("Model failed")
        mock_model_class.from_pretrained.return_value = mock_model

        mock_config.validate()
        
        # Initialize model
        model = AdvancedModel(mock_config.model_variant, mock_config.device)
        
        # Test health check with failure
        health = model.health_check()
        
        assert not health.is_healthy
        assert health.error_count > 0
        assert "Health check failed" in health.last_error

    def test_config_validation(self, temp_dir):
        """Test configuration validation with various scenarios."""
        # Test valid config
        valid_config = Config(
            data_path=temp_dir,
            output_dir=temp_dir,
            batch_size=1,
            max_seq_len=128
        )
        valid_config.validate()  # Should not raise
        
        # Test invalid batch size
        invalid_config = Config(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            invalid_config.validate()
        
        # Test invalid max_seq_len
        invalid_config = Config(max_seq_len=0)
        with pytest.raises(ValueError, match="max_seq_len must be a positive integer"):
            invalid_config.validate()

    def test_config_file_operations(self, temp_dir):
        """Test configuration file save/load operations."""
        config = Config(
            task='MATH',
            data_path=temp_dir,
            output_dir=temp_dir,
            batch_size=2,
            learning_rate=1e-3
        )
        
        config_file = os.path.join(temp_dir, 'test_config.json')
        
        # Test save
        config.save_to_file(config_file)
        assert os.path.exists(config_file)
        
        # Test load
        loaded_config = Config.from_file(config_file)
        assert loaded_config.task == config.task
        assert loaded_config.batch_size == config.batch_size
        assert loaded_config.learning_rate == config.learning_rate

    def test_data_loading_edge_cases(self, temp_dir):
        """Test data loading with edge cases."""
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Test empty file
        empty_file = os.path.join(data_dir, 'empty.json')
        with open(empty_file, 'w') as f:
            f.write('')
        
        data = load_json(empty_file)
        assert data == []
        
        # Test invalid JSON
        invalid_file = os.path.join(data_dir, 'invalid.json')
        with open(invalid_file, 'w') as f:
            f.write('invalid json content')
        
        with pytest.raises(ValueError, match="Invalid JSON format"):
            load_json(invalid_file)
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            load_json('non_existent.json')

    def test_memory_management(self, mock_config):
        """Test memory management features."""
        with patch('src.score_model.model.LlamaForCausalLM'), \
             patch('src.score_model.model.LlamaTokenizer') as mock_tokenizer_class:
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = '[PAD]'
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_config.validate()
            
            model = AdvancedModel(mock_config.model_variant, mock_config.device)
            
            # Test memory usage tracking
            memory_stats = model.get_memory_usage()
            assert isinstance(memory_stats, dict)
            
            # Test cache clearing
            model.clear_cache()  # Should not raise

    @patch('src.score_model.model.LlamaForCausalLM')
    @patch('src.score_model.model.LlamaTokenizer')
    def test_model_save_load_cycle(self, mock_tokenizer_class, mock_model_class, 
                                  mock_config, temp_dir):
        """Test model save and load cycle."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = '[PAD]'
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.save_pretrained = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_config.validate()
        
        # Initialize model
        model = AdvancedModel(mock_config.model_variant, mock_config.device)
        
        # Test save
        save_path = os.path.join(temp_dir, 'test_model')
        model.save_pretrained(save_path)
        
        # Verify save was called
        mock_model.save_pretrained.assert_called_once_with(save_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(save_path)

    def test_logging_system(self, mock_config):
        """Test the structured logging system."""
        mock_config.validate()
        
        logger = setup_logging(mock_config)
        
        # Test basic logging
        logger.info("Test message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Test structured logging
        test_metrics = {'test_key': 'test_value', 'timestamp': 12345}
        logger.log_metrics(test_metrics)
        
        # Verify log files exist
        log_file = mock_config.get_log_file_path('score_training')
        assert os.path.exists(log_file)
        
        metrics_file = Path(mock_config.log_dir) / 'score_training_metrics.jsonl'
        assert metrics_file.exists()

if __name__ == '__main__':
    pytest.main([__file__]) 