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

def order_averaged_scores(unordered_scores: List[Dict[str, Any]], objective_fn: Callable[[List[float]], float]) -> List[tuple[str, List[ModelParameters], float]]:
    """
    Order model results by their objective function score.
    
    Args:
        unordered_scores: List of model analysis results
        objective_fn: Function to calculate final score from validation metrics
        
    Returns:
        List of (model_name, parameters, score) tuples, sorted by score
    """
    model_param_scores = {}
    
    # Group scores by model and parameters
    for result in unordered_scores:
        model_results = result["model_results"]
        parameters = result["parameters"]
        
        for model_name, metrics in model_results.items():
            score = metrics['val_metrics']['MAE']  # Using MAE from validation metrics
            
            key = (model_name, tuple(parameters))
            if key not in model_param_scores:
                model_param_scores[key] = []
            model_param_scores[key].append(score)
    
    # Calculate final scores using objective function
    final_scores = [
        (model_name, list(params), objective_fn(scores))
        for (model_name, params), scores in model_param_scores.items()
    ]
    
    # Sort by score
    return sorted(final_scores, key=lambda x: x[2])

def search_models_and_parameters(
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
    results = []
    
    for parameter_combo in parameter_combinations:

        # Update model features
        for model in models.values():
            model.update_features(parameter_combo)

        for interval in time_intervals:

            # Run model analysis for this parameter combination and time interval
            run_model_results = run_model_analysis(models, train_data, mode, interval, val_data)
            res = {"model_results": run_model_results, "parameters": parameter_combo, "time_interval": interval}
            results.append(res)
            # for (name, model) in models.items():
            #     update_results(results, name, parameter_combo, res[name]['val_metrics']['MAE'])
        
    return results