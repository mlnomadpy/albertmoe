"""
Evaluation utilities for AlbertMoE models.
"""

import os
import wandb
from mteb import MTEB, get_tasks

from .models import SentenceAlbert
from .config import AlbertMoEConfig


class MTEBEvaluator:
    """MTEB evaluation wrapper for AlbertMoE models."""
    
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.mteb_model = SentenceAlbert(config, model_path)
    
    def evaluate(self, eval_tasks="basic", custom_tasks=None, output_folder=None):
        """
        Evaluate model on MTEB tasks.
        
        Args:
            eval_tasks: "basic" for STSBenchmark only, "comprehensive" for all tasks
            custom_tasks: List of custom task names
            output_folder: Output folder for results
        """
        if eval_tasks == "basic":
            task_list = ["STSBenchmark"]
        elif eval_tasks == "comprehensive":
            task_list = [t.metadata.name for t in get_tasks()]
        else:
            task_list = custom_tasks or ["STSBenchmark"]
            
        tasks = get_tasks(tasks=task_list)
        evaluation = MTEB(tasks=tasks)
        
        if output_folder is None:
            output_folder = f"results/{os.path.basename(self.model_path)}_{eval_tasks}"
        
        results = evaluation.run(self.mteb_model, output_folder=output_folder)
        return results
    
    def log_results_to_wandb(self, results, task_type="evaluation"):
        """Log MTEB results to Weights & Biases."""
        all_logged_metrics = {}
        
        for task_result in results:
            task_name = task_result.task_name
            for split_name, scores in task_result.scores.items():
                if isinstance(scores, list):
                    for i, score_dict in enumerate(scores):
                        for metric_name, value in score_dict.items():
                            if isinstance(value, (int, float)):
                                key = f"eval/{task_name}/{split_name}_{i}/{metric_name}"
                                all_logged_metrics[key] = value
                elif isinstance(scores, dict):
                    if any(isinstance(v, dict) for v in scores.values()):
                        for main_metric, metric_dict in scores.items():
                            for metric_name, value in metric_dict.items():
                                if isinstance(value, (int, float)):
                                    all_logged_metrics[f"eval/{task_name}/{split_name}/{main_metric}_{metric_name}"] = value
                    else:
                        for metric_name, value in scores.items():
                            if isinstance(value, (int, float)):
                                all_logged_metrics[f"eval/{task_name}/{split_name}/{metric_name}"] = value
        
        if all_logged_metrics and wandb.run is not None:
            wandb.log(all_logged_metrics)
            print(f"✅ Logged {len(all_logged_metrics)} metrics to W&B.")
        
        return all_logged_metrics


def evaluate_model(model_path, task_type="clm", eval_tasks="basic", custom_tasks=None, use_wandb=False, wandb_project=None):
    """
    Convenience function to evaluate a trained model.
    
    Args:
        model_path: Path to the trained model
        task_type: "clm" or "mlm" 
        eval_tasks: "basic" or "comprehensive"
        custom_tasks: List of custom MTEB task names
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
    """
    # Create default config (this should ideally be loaded from the model)
    config = AlbertMoEConfig()
    
    # Initialize W&B if requested
    if use_wandb and wandb.run is None:
        wandb.init(
            project=wandb_project or "albert-moe-eval",
            name=f"evaluate-{os.path.basename(model_path)}",
            config={
                "model_path": model_path,
                "task_type": task_type,
                "eval_tasks": eval_tasks
            }
        )
    
    # Create evaluator and run evaluation
    evaluator = MTEBEvaluator(model_path, config)
    results = evaluator.evaluate(eval_tasks, custom_tasks)
    
    # Log results if W&B is enabled
    if use_wandb:
        evaluator.log_results_to_wandb(results)
    
    if wandb.run is not None:
        wandb.finish()
    
    return results