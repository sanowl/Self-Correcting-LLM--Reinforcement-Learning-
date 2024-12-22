#!/usr/bin/env python3
"""
Script for model inference.
"""

import os
import json
import torch
import argparse
import logging
from typing import List, Dict, Any

from src.score_model.config import Config
from src.score_model.model import AdvancedModel
from src.score_model.utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_questions(input_file: str) -> List[Dict[str, Any]]:
    """
    Load questions from a file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                return [json.loads(line) for line in f if line.strip()]
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        raise

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save results to a file.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def run_inference(
    model: AdvancedModel,
    questions: List[Dict[str, Any]],
    config: Config,
    max_attempts: int = 2
) -> List[Dict[str, Any]]:
    """
    Run inference on a list of questions.
    """
    results = []
    
    for question in questions:
        try:
            # First attempt
            input_text = question['question'] if config.task == 'MATH' else question['text']
            encodings = model.tokenizer(
                input_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=config.max_seq_len
            ).to(config.device)
            
            output_ids = model.generate_text(encodings, temperature=0.7)
            first_attempt = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Second attempt (if needed)
            if max_attempts > 1:
                second_input = f"{input_text}\nPrevious Attempt:\n{first_attempt}\nInstructions: Please correct the above attempt."
                second_encodings = model.tokenizer(
                    second_input,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=config.max_seq_len
                ).to(config.device)
                
                output_ids = model.generate_text(second_encodings, temperature=0.7)
                second_attempt = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                second_attempt = None
            
            # Store results
            result = {
                'question': input_text,
                'first_attempt': first_attempt,
                'second_attempt': second_attempt
            }
            if 'answer' in question:
                result['correct_answer'] = question['answer']
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing question '{input_text[:50]}...': {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference using trained model")
    parser.add_argument('--model-path', type=str, required=True,
                      help="Path to the trained model")
    parser.add_argument('--model-variant', type=str,
                      default='decapoda-research/llama-7b-hf',
                      help="Model variant identifier")
    parser.add_argument('--input', type=str, required=True,
                      help="Input file with questions")
    parser.add_argument('--output', type=str, required=True,
                      help="Output file for results")
    parser.add_argument('--task', type=str, choices=['MATH', 'CODE'],
                      default='MATH', help="Task type")
    parser.add_argument('--max-attempts', type=int, default=2,
                      help="Maximum number of attempts per question")
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = Config(
            task=args.task,
            model_variant=args.model_variant
        )
        
        # Load model
        model = load_model(args.model_path, args.model_variant, config.device)
        logger.info("Model loaded successfully")
        
        # Load questions
        questions = load_questions(args.input)
        logger.info(f"Loaded {len(questions)} questions")
        
        # Run inference
        results = run_inference(model, questions, config, args.max_attempts)
        logger.info(f"Processed {len(results)} questions")
        
        # Save results
        save_results(results, args.output)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        exit(1)

if __name__ == '__main__':
    main() 