import logging
from typing import Any, Dict, List
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """
    Base dataset class for loading data.
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            return self.data[idx]
        except IndexError as e:
            logger.error(f"Index {idx} out of range for dataset of size {len(self.data)}.")
            raise IndexError("Dataset index out of range.") from e
        except Exception as e:
            logger.error(f"Error retrieving item at index {idx}: {e}")
            raise 