import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union


def plot_anomaly_scores(
    scores: Union[List[float], np.ndarray], 
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None
) -> Tuple[str, str]:
    """
    Plot and save anomaly scores.
    
    Args:
        scores (list or np.ndarray): List of anomaly scores to plot
        config (dict): Configuration dictionary
        output_dir (str, optional): Directory to save the plot and data.
            Defaults to config's output_dir/anomaly_plots.
        filename_prefix (str, optional): Prefix for output filenames.
            Defaults to model name from config.
    
    Returns:
        tuple: Paths to the saved plot and data files (plot_path, data_path)
    """
    if not scores:
        print("No anomaly scores to plot.")
        return None, None
        
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(config.get("output_dir", "outputs"), "anomaly_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup filename prefix
    if filename_prefix is None:
        filename_prefix = config.get("which_model", "model")
    
    # Get plotting configuration
    plot_config = config.get("anomaly_plot_config", {})
    figsize = plot_config.get("figsize", (12, 6))
    line_color = plot_config.get("color", "blue")
    threshold_color = plot_config.get("threshold_color", "red")
    grid = plot_config.get("grid", True)
    dpi = plot_config.get("dpi", 300)
    
    # Plot anomaly scores
    plt.figure(figsize=figsize)
    plt.plot(scores, color=line_color, label='Anomaly Score')
    
    # Add threshold if specified
    threshold = config.get("anomaly_threshold")
    if threshold is not None:
        plt.axhline(y=threshold, color=threshold_color, linestyle='--', label=f'Threshold ({threshold})')
    
    # Add labels and title
    model_name = config.get("which_model", "Model")
    loss_fn = config.get("loss_fn", "Unknown Loss")
    plt.title(f'Anomaly Scores for {model_name} Model using {loss_fn}')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.legend()
    
    if grid:
        plt.grid(True)
    
    # Save the figure
    plot_path = os.path.join(output_dir, f"{filename_prefix}_anomaly_scores.png")
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Anomaly score plot saved to {plot_path}")
    
    # Save the raw scores as numpy array for future analysis
    data_path = os.path.join(output_dir, f"{filename_prefix}_anomaly_scores.npy")
    np.save(data_path, np.array(scores))
    print(f"Anomaly scores saved to {data_path}")
    
    return plot_path, data_path


def calculate_threshold(
    scores: Union[List[float], np.ndarray],
    method: str = 'percentile',
    percentile: float = 95.0,
    std_multiplier: float = 3.0
) -> float:
    """
    Calculate an anomaly detection threshold using different methods.
    
    Args:
        scores (list or np.ndarray): List of anomaly scores
        method (str): Method to calculate threshold ('percentile', 'std', 'iqr')
        percentile (float): Percentile for threshold (if method='percentile')
        std_multiplier (float): Multiplier for standard deviation (if method='std')
        
    Returns:
        float: Calculated threshold value
    """
    scores_array = np.array(scores)
    
    if method == 'percentile':
        # Percentile-based threshold (e.g., 95th percentile)
        return float(np.percentile(scores_array, percentile))
    
    elif method == 'std':
        # Mean + N*std threshold
        mean = np.mean(scores_array)
        std = np.std(scores_array)
        return float(mean + std_multiplier * std)
    
    elif method == 'iqr':
        # IQR-based threshold (Q3 + 1.5*IQR)
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1
        return float(q3 + 1.5 * iqr)
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def detect_anomalies(
    scores: Union[List[float], np.ndarray],
    threshold: Optional[float] = None,
    threshold_method: str = 'percentile'
) -> Tuple[np.ndarray, float]:
    """
    Detect anomalies in scores based on threshold.
    
    Args:
        scores (list or np.ndarray): List of anomaly scores
        threshold (float, optional): Threshold value. If None, it will be calculated
        threshold_method (str): Method to calculate threshold if not provided
        
    Returns:
        tuple: (anomaly_indices, threshold_used)
    """
    scores_array = np.array(scores)
    
    # Calculate threshold if not provided
    if threshold is None:
        threshold = calculate_threshold(scores_array, method=threshold_method)
    
    # Detect anomalies
    anomaly_indices = np.where(scores_array > threshold)[0]
    
    return anomaly_indices, threshold