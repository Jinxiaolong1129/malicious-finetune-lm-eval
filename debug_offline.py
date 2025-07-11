#!/usr/bin/env python3
"""
lm-eval SimpleEvaluate debug script for evaluating model on WinoGrande task
Converted from lm_eval command line interface
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lm_eval_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main evaluation function using lm-eval SimpleEvaluate"""
    try:
        # Import from lm_eval
        from lm_eval import simple_evaluate
        logger.info("Successfully imported lm_eval.simple_evaluate")
        
    except ImportError as e:
        logger.error(f"Failed to import lm_eval.simple_evaluate: {e}")
        logger.info("Please install lm_eval: pip install lm_eval")
        return
    
    # Model configuration
    model_path = "/data3/user/jin509/hf_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    
    # Verify model path exists
    if not Path(model_path).exists():
        logger.error(f"Model path does not exist: {model_path}")
        return
    
    logger.info(f"Using model path: {model_path}")
    
    # Define evaluation parameters
    model_args = {
        'pretrained': model_path,
        'device': 'cuda:0'
    }
    tasks = ['winogrande']
    
    # tasks = ['truthfulqa_mc1']
    # tasks = ["humaneval"]
    
    
    # Run evaluation
    try:
        logger.info(f"Starting evaluation on tasks: {tasks}")
        logger.info(f"Model args: {model_args}")
        
        # Call simple_evaluate function
        results = simple_evaluate(
            model="hf",  # model type
            model_args=model_args,  # model arguments
            tasks=tasks,  # tasks to evaluate
            confirm_run_unsafe_code=True,
            # Optional parameters you can add:
            # num_fewshot=0,  # number of few-shot examples
            # batch_size=8,  # batch size for evaluation
            # limit=None,  # limit number of examples
            # device=None,  # device will be set via model_args
            # use_cache=None,  # use cached results
            # cache_requests=False,  # cache requests
            # write_out=False,  # write out results
            # log_samples=False,  # log individual samples
            # verbosity="INFO",  # verbosity level
        )
        
        logger.info("Evaluation completed successfully")
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Print task results
        if 'results' in results:
            for task, result in results['results'].items():
                print(f"\nTask: {task}")
                print("-" * 20)
                for metric, value in result.items():
                    print(f"{metric}: {value}")
        else:
            print("Results:")
            print(results)
        
        # Print configuration info
        if 'config' in results:
            print(f"\nConfiguration:")
            print("-" * 20)
            config = results['config']
            print(f"Model: {config.get('model', 'N/A')}")
            print(f"Model args: {config.get('model_args', 'N/A')}")
            print(f"Batch size: {config.get('batch_size', 'N/A')}")
            print(f"Device: {config.get('device', 'N/A')}")
        
        # Save results to file
        results_file = "lm_eval_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Also save a human-readable summary
        summary_file = "lm_eval_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("lm-eval Evaluation Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: hf\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Tasks: {tasks}\n")
            f.write(f"Device: cuda:1\n\n")
            
            if 'results' in results:
                for task, result in results['results'].items():
                    f.write(f"Task: {task}\n")
                    f.write("-" * 20 + "\n")
                    for metric, value in result.items():
                        f.write(f"{metric}: {value}\n")
                    f.write("\n")
        
        logger.info(f"Summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.debug("Stack trace:", exc_info=True)
        print(e)
        # Print additional debugging info
        print("\nDEBUG INFORMATION:")
        print("-" * 30)
        print(f"Python version: {sys.version}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model path exists: {Path(model_path).exists()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        
        return

if __name__ == "__main__":
    logger.info("Starting lm-eval SimpleEvaluate debug script")
    main()
    logger.info("Script completed")