"""
Script for analyzing and visualizing anomaly scores from trained models.
Can be used to load saved anomaly scores or run a model on test data to generate scores.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer

# Ensure the modules directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from src.data_module import DataModule
from src.lightning_module import LightningModel
from src.anomaly_visualization import plot_anomaly_scores, detect_anomalies, calculate_threshold


def load_scores(file_path):
    """Load anomaly scores from a numpy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Score file not found: {file_path}")
    
    return np.load(file_path)


def generate_scores(model_path, config):
    """Generate anomaly scores by running a model on test data."""
    # Initialize modules
    data_module = DataModule(config)
    
    # Initialize the appropriate model architecture
    if config['which_model'] == 'CAE':
        from models.cae import CAEModel
        base_model = CAEModel(config)
    elif config['which_model'] == 'WavenetAE':
        from models.wavenet_ae import WaveNetAEModel
        base_model = WaveNetAEModel(config)
    elif config['which_model'] == 'AttentionAE':
        from models.attention_ae import AttentionAEModel
        base_model = AttentionAEModel(config)
    else:
        raise ValueError(f"Unknown model type: {config['which_model']}")
    
    # Load the trained model
    model = LightningModel.load_from_checkpoint(model_path, model=base_model, config=config)
    model.eval()
    
    # Set up trainer with no logging
    trainer = Trainer(accelerator="cpu", logger=False)
    
    # Run test
    test_results = trainer.test(model, datamodule=data_module)
    
    # Return collected scores
    return np.array(model.test_anomaly_scores)


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize anomaly scores")
    parser.add_argument("--scores", type=str, help="Path to saved anomaly scores (.npy file)")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="anomaly_analysis", help="Output directory")
    parser.add_argument("--threshold_method", type=str, default="percentile", 
                        choices=["percentile", "std", "iqr"], help="Method to calculate threshold")
    args = parser.parse_args()
    
    # Either load existing scores or generate new ones
    if args.scores:
        print(f"Loading anomaly scores from {args.scores}")
        scores = load_scores(args.scores)
    elif args.model:
        print(f"Generating anomaly scores using model {args.model}")
        scores = generate_scores(args.model, config)
    else:
        parser.error("Either --scores or --model must be provided")
    
    # Calculate threshold
    threshold = calculate_threshold(scores, method=args.threshold_method)
    print(f"Calculated threshold ({args.threshold_method}): {threshold:.4f}")
    
    # Detect anomalies
    anomaly_indices, _ = detect_anomalies(scores, threshold=threshold)
    print(f"Found {len(anomaly_indices)} anomalies out of {len(scores)} samples "
          f"({len(anomaly_indices)/len(scores)*100:.2f}%)")
    
    # Update config for plotting
    analysis_config = config.copy()
    analysis_config.update({
        "output_dir": args.output,
        "anomaly_threshold": threshold,
        "which_model": os.path.basename(args.model or args.scores).split("_")[0]
    })
    
    # Plot scores with threshold
    plot_path, _ = plot_anomaly_scores(
        scores,
        config=analysis_config,
        filename_prefix=f"{analysis_config['which_model']}_{args.threshold_method}_threshold"
    )
    
    print(f"Analysis complete. Plot saved to {plot_path}")
    
    # Save anomaly indices for further analysis
    indices_path = os.path.join(args.output, "anomaly_indices.npy")
    np.save(indices_path, anomaly_indices)
    print(f"Anomaly indices saved to {indices_path}")


if __name__ == "__main__":
    main()


    
# Analyze previously saved scores
# python scripts/analyze_anomalies.py --scores outputs/anomaly_plots/CAE_anomaly_scores.npy --threshold_method percentile

# Generate and analyze scores using a trained model
# python scripts/analyze_anomalies.py --model model_checkpoints/model-best.ckpt --threshold_method std