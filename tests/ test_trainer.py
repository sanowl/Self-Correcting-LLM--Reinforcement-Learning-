# tests/test_utils.py

import os
import json
import unittest
import tempfile
from unittest import mock
from your_module import set_seed, load_json
from hypothesis import given, strategies as st

class TestUtils(unittest.TestCase):
    def test_set_seed(self):
        """Test if the seed is set correctly by generating random numbers."""
        with mock.patch('random.seed') as mock_random_seed, \
             mock.patch('numpy.random.seed') as mock_numpy_seed, \
             mock.patch('torch.manual_seed') as mock_torch_seed:
            set_seed(123)
            mock_random_seed.assert_called_once_with(123)
            mock_numpy_seed.assert_called_once_with(123)
            mock_torch_seed.assert_called_once_with(123)

    @given(st.lists(st.dictionaries(keys=st.text(min_size=1), values=st.integers()), min_size=1, max_size=100))
    def test_load_json_jsonl_all_samples(self, data):
        """Test loading all samples from a JSONL file using property-based testing."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as tmp:
            for item in data:
                tmp.write(json.dumps(item) + "\n")
            tmp_path = tmp.name

        try:
            loaded_data = load_json(tmp_path)
            self.assertEqual(loaded_data, data)
        finally:
            os.remove(tmp_path)

    @given(st.lists(st.dictionaries(keys=st.text(min_size=1), values=st.integers()), min_size=1, max_size=100))
    def test_load_json_jsonl_with_max_samples(self, data):
        """Test loading a limited number of samples from a JSONL file."""
        max_samples = len(data) // 2
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as tmp:
            for item in data:
                tmp.write(json.dumps(item) + "\n")
            tmp_path = tmp.name

        try:
            loaded_data = load_json(tmp_path, max_samples=max_samples)
            self.assertEqual(loaded_data, data[:max_samples])
        finally:
            os.remove(tmp_path)

    @given(st.lists(st.dictionaries(keys=st.text(min_size=1), values=st.integers()), min_size=1, max_size=100))
    def test_load_json_json_all_samples(self, data):
        """Test loading all samples from a JSON file using property-based testing."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
            json.dump({"items": data}, tmp)
            tmp_path = tmp.name

        try:
            loaded_data = load_json(tmp_path)
            self.assertEqual(loaded_data, data)
        finally:
            os.remove(tmp_path)

    @given(st.integers())
    def test_set_seed_with_random_integers(self, seed):
        """Property-based test to ensure set_seed accepts any integer."""
        with mock.patch('random.seed') as mock_random_seed, \
             mock.patch('numpy.random.seed') as mock_numpy_seed, \
             mock.patch('torch.manual_seed') as mock_torch_seed:
            set_seed(seed)
            mock_random_seed.assert_called_once_with(seed)
            mock_numpy_seed.assert_called_once_with(seed)
            mock_torch_seed.assert_called_once_with(seed)

    def test_load_json_invalid_file(self):
        """Test loading from a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_json("non_existent_file.json")

    def test_load_json_invalid_json(self):
        """Test loading from an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
            tmp.write("Invalid JSON content")
            tmp_path = tmp.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                load_json(tmp_path)
        finally:
            os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
