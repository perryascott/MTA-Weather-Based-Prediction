from itertools import combinations
from typing import List, Callable, Dict, Any
from dataclasses import asdict
import numpy as np
from scripts.model import ModelParameters, run_model_analysis
from scripts.filter import TimeInterval

def mae_objective(scores: List[float]) -> float:
    """Mean Absolute Error objective function"""
    return np.mean(np.abs(scores))

def mse_objective(scores: List[float]) -> float:
    """Mean Squared Error objective function"""
    return np.mean([s**2 for s in scores])

def generate_feature_combinations(params: List[ModelParameters]) -> List[List[ModelParameters]]:
    """
    Generate all possible combinations of features from ModelParameters.
    Returns a list of lists of ModelParameters, each containing at least one feature.
    """
    
    all_combinations = []
    # Generate combinations of different lengths (1 to max features)
    for r in range(1, len(params) + 1):
        for combo in combinations(params, r):
            # Create list of selected ModelParameters
            all_combinations.append(list(combo))
    
    return all_combinations

def update_results(results, model_name, params, score):
    if (model_name, tuple(params)) in results:
        results[(model_name, tuple(params))].append(score)
    else:
        results[(model_name, tuple(params))] = [score]

def evaluate_models(
    models,
    train_data,
    val_data,
    mode,
    time_intervals: List[TimeInterval],
    parameter_combinations: List[List[ModelParameters]],
    objective_fn: Callable[[List[float]], float] = mae_objective
) -> List[tuple[List[ModelParameters], float]]:
    """
    Evaluate models with different parameter combinations using the specified objective function.
    Returns a sorted list of (parameters, score) tuples, best performing first.
    
    Args:
        time_intervals: List of TimeInterval objects containing validation results
        parameter_combinations: List of parameter combinations to evaluate
        objective_fn: Function that takes a list of validation scores and returns a single score
    """
    results = {}
    
    for params in parameter_combinations:
        # Collect validation scores across all time intervals for this parameter combination
        validation_scores = []
        for interval in time_intervals:

            # Run model analysis for this parameter combination and time interval
            res = run_model_analysis(models, train_data, mode, interval, val_data)
            for (name, model) in models.items():
                update_results(results, name, params, res[name]['val_metrics']['MAE'])
        
    
    model_param_scores = {}
    for (model_name, params), scores in results.items():
        model_param_scores[model_name, params] = objective_fn(scores)
    
    # Sort scores
    sorted_scores = sorted(model_param_scores.items(), key=lambda x: x[1])
    
    return sorted_scores