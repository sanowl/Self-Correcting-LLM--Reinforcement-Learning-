#!/usr/bin/env python3
"""
Script to preprocess and validate training data.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any
from sympy import simplify, SympifyError
from sympy.parsing.sympy_parser import parse_expr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def validate_math_sample(sample: Dict[str, Any]) -> bool:
    """
    Validate a math sample.
    """
    required_keys = ['question', 'answer']
    if not all(key in sample for key in required_keys):
        return False
    
    try:
        # Try to parse the answer as a mathematical expression
        parse_expr(str(sample['answer']))
        return True
    except (SympifyError, TypeError):
        return False
    except Exception as e:
        logger.warning(f"Unexpected error validating math sample: {e}")
        return False

def validate_code_sample(sample: Dict[str, Any]) -> bool:
    """
    Validate a code sample.
    """
    required_keys = ['text', 'code', 'test']
    if not all(key in sample for key in required_keys):
        return False
    
    try:
        # Basic syntax check
        compile(sample['code'], '<string>', 'exec')
        compile(sample['test'], '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception as e:
        logger.warning(f"Unexpected error validating code sample: {e}")
        return False

def process_dataset(
    input_file: str,
    output_file: str,
    task: str,
    max_samples: int = None
) -> None:
    """
    Process and validate a dataset.
    """
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        
        logger.info(f"Loaded {len(data)} samples from {input_file}")
        
        # Validate samples
        validator = validate_math_sample if task == 'MATH' else validate_code_sample
        valid_samples = []
        for sample in data:
            if validator(sample):
                valid_samples.append(sample)
            if max_samples and len(valid_samples) >= max_samples:
                break
        
        logger.info(f"Found {len(valid_samples)} valid samples")
        
        # Write output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, indent=2)
        
        logger.info(f"Saved processed data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Preprocess and validate training data")
    parser.add_argument('--input', type=str, required=True, help="Input data file")
    parser.add_argument('--output', type=str, required=True, help="Output data file")
    parser.add_argument('--task', type=str, required=True, choices=['MATH', 'CODE'],
                      help="Task type (MATH or CODE)")
    parser.add_argument('--max-samples', type=int, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    try:
        process_dataset(args.input, args.output, args.task, args.max_samples)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)

if __name__ == '__main__':
    main() 