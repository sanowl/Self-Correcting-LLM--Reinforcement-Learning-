import sys
import os
import json
import unittest
import tempfile
import logging
from unittest import mock
from hypothesis import given, strategies as st

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import set_seed, load_json

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test resources."""
        logger.info("Setting up test.")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Tearing down test.")
    
    def test_set_seed(self):
        """Test if the seed is set correctly by generating random numbers."""
        logger.debug("Testing set_seed with fixed seed 123.")
        with mock.patch('random.seed') as mock_random_seed, \
             mock.patch('numpy.random.seed') as mock_numpy_seed, \
             mock.patch('torch.manual_seed') as mock_torch_seed:  # Corrected typo here
            set_seed(123)
            mock_random_seed.assert_called_once_with(123)
            mock_numpy_seed.assert_called_once_with(123)
            mock_torch_seed.assert_called_once_with(123)
        logger.debug("set_seed test passed for seed 123.")

    @given(st.lists(st.dictionaries(keys=st.text(min_size=1), values=st.integers()), min_size=1, max_size=100))
    def test_load_json_jsonl_all_samples(self, data):
        """Test loading all samples from a JSONL file using property-based testing."""
        logger.debug("Testing load_json_jsonl_all_samples with data of length %s", len(data))
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as tmp:
            for item in data:
                tmp.write(json.dumps(item) + "\n")
            tmp_path = tmp.name

        try:
            logger.debug("Calling load_json on JSONL file with all samples.")
            loaded_data = load_json(tmp_path)
            logger.debug("Loaded data: %s", loaded_data)
            self.assertEqual(loaded_data, data)
        finally:
            os.remove(tmp_path)
            logger.debug("Temporary JSONL file removed.")

    @given(st.lists(st.dictionaries(keys=st.text(min_size=1), values=st.integers()), min_size=1, max_size=100))
    def test_load_json_jsonl_with_max_samples(self, data):
        """Test loading a limited number of samples from a JSONL file."""
        max_samples = max(1, len(data) // 2)  # Ensure max_samples is at least 1
        logger.debug("Testing load_json_jsonl_with_max_samples with max_samples %d", max_samples)
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as tmp:
            for item in data:
                tmp.write(json.dumps(item) + "\n")
            tmp_path = tmp.name

        try:
            logger.debug("Calling load_json with max_samples %d", max_samples)
            loaded_data = load_json(tmp_path, max_samples=max_samples)
            logger.debug("Loaded data: %s", loaded_data)
            self.assertEqual(loaded_data, data[:max_samples])
        finally:
            os.remove(tmp_path)
            logger.debug("Temporary JSONL file removed.")

    @given(st.lists(st.dictionaries(keys=st.text(min_size=1), values=st.integers()), min_size=1, max_size=100))
    def test_load_json_json_all_samples(self, data):
        """Test loading all samples from a JSON file using property-based testing."""
        logger.debug("Testing load_json_json_all_samples with data of length %s", len(data))
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name

        try:
            logger.debug("Calling load_json on JSON file with all samples.")
            loaded_data = load_json(tmp_path)
            logger.debug("Loaded data: %s", loaded_data)
            self.assertEqual(loaded_data, data)
        finally:
            os.remove(tmp_path)
            logger.debug("Temporary JSON file removed.")

    @given(st.integers())
    def test_set_seed_with_random_integers(self, seed):
        """Property-based test to ensure set_seed accepts any integer."""
        logger.debug("Testing set_seed with random seed: %s", seed)
        with mock.patch('random.seed') as mock_random_seed, \
             mock.patch('numpy.random.seed') as mock_numpy_seed, \
             mock.patch('torch.manual_seed') as mock_torch_seed:
            set_seed(seed)
            mock_random_seed.assert_called_once_with(seed)
            mock_numpy_seed.assert_called_once_with(seed)
            mock_torch_seed.assert_called_once_with(seed)
        logger.debug("set_seed test passed for random seed: %s", seed)

    def test_load_json_invalid_file(self):
        """Test loading from a non-existent file."""
        logger.debug("Testing load_json on a non-existent file.")
        with self.assertRaises(FileNotFoundError):
            load_json("non_existent_file.json")
        logger.debug("FileNotFoundError raised as expected.")

    def test_load_json_invalid_json(self):
        """Test loading from an invalid JSON file."""
        logger.debug("Testing load_json on an invalid JSON file.")
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
            tmp.write("Invalid JSON content")
            tmp_path = tmp.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                load_json(tmp_path)
            logger.debug("JSONDecodeError raised as expected.")
        finally:
            os.remove(tmp_path)
            logger.debug("Temporary invalid JSON file removed.")


if __name__ == '__main__':
    unittest.main()
