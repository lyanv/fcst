import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Calculate Root Mean Squared Scaled Error."""
    naive_forecast = y_train[:-1]
    naive_errors = y_train[1:] - naive_forecast
    scale = np.mean(naive_errors ** 2)
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse / (scale + 1e-8))


def evaluate_mcc_forecasting(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    """
    Evaluate MCC aggregates forecasting performance.
    
    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)
        y_train: Training data for RMSSE calculation, shape (n_train_samples,)
    
    Returns:
        Dictionary with metrics:
        - 'sMAPE_w': Weekly symmetric MAPE
        - 'RMSSE_w': Weekly RMSSE
        - 'MAE': Mean Absolute Error
        - 'RMSE': Root Mean Squared Error
    """
    metrics = {
        'sMAPE_w': smape(y_true, y_pred),
        'RMSSE_w': rmsse(y_true, y_pred, y_train),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
    return metrics


def generate_mcc_report(model_name: str, overall_metrics: Dict[str, float],
                       category_metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Generate unified report for MCC aggregates forecasting model.
    
    Args:
        model_name: Name of the model
        overall_metrics: Overall performance metrics
        category_metrics: Per-category performance metrics
    
    Returns:
        Formatted report string
    """
    report = f"\n{'='*60}\n"
    report += f"MCC AGGREGATES FORECASTING REPORT: {model_name.upper()}\n"
    report += f"{'='*60}\n\n"
    
    # Overall metrics
    report += "OVERALL PERFORMANCE:\n"
    report += f"{'Metric':<15} {'Value':<15}\n"
    report += f"{'-'*30}\n"
    for metric, value in overall_metrics.items():
        report += f"{metric:<15} {value:<15.4f}\n"
    
    # Category-wise metrics
    if category_metrics:
        report += f"\nPER-CATEGORY PERFORMANCE:\n"
        categories = list(category_metrics.keys())
        metrics_names = list(category_metrics[categories[0]].keys())
        
        # Header
        header = f"{'Category':<15}"
        for metric in metrics_names:
            header += f"{metric:<12}"
        report += header + "\n"
        report += f"{'-'*len(header)}\n"
        
        # Data rows
        for category in sorted(categories):
            row = f"{category:<15}"
            for metric in metrics_names:
                value = category_metrics[category][metric]
                row += f"{value:<12.4f}"
            report += row + "\n"
    
    report += f"\n{'='*60}\n"
    return report


def evaluate_and_report_mcc(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_train: np.ndarray, categories: Optional[np.ndarray] = None,
                          print_report: bool = True) -> Tuple[Dict[str, float], str]:
    """
    Complete evaluation and reporting for MCC forecasting model.
    
    Args:
        model_name: Name of the model
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for RMSSE
        categories: Category labels (optional)
        print_report: Whether to print the report
    
    Returns:
        Tuple of (overall_metrics, report_string)
    """
    # Overall metrics
    overall_metrics = evaluate_mcc_forecasting(y_true, y_pred, y_train)
    
    # Category-wise metrics
    category_metrics = {}
    if categories is not None:
        unique_categories = np.unique(categories)
        for cat in unique_categories:
            mask = categories == cat
            if np.sum(mask) > 0:
                cat_metrics = evaluate_mcc_forecasting(
                    y_true[mask], y_pred[mask], y_train
                )
                category_metrics[cat] = cat_metrics
    
    # Generate report
    report = generate_mcc_report(model_name, overall_metrics, category_metrics)
    
    if print_report:
        print(report)
    
    return overall_metrics, report 