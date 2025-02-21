# Model and Training Configuration
config = {
    #######################
    # General Parameters  #
    #######################
    "which_model": "CAE",  # available options: CAE, WavenetAE, AttentionAE
    "frame_length": 4096,
    "hop_length": 2048,
    "batch_size": 32,
    "random_seed": 21,
    
    #######################
    # Dataset Parameters  #
    #######################
    "train_val_ratio": 0.8,
    "is_aug": True,
    "preprocess_pipe": [
        "SplitToFrame() +"
        "Normalize() +"
        "Augmentation() for train data if configured"
    ],
    
    #######################
    # CAE Model Params    #
    #######################
    "dropout_input": 0.4,
    "dropout_hidden": 0.2,
    "encoder_layer_num": 3,
    "conv_kernel_sizes": [21, 11, 11],
    "stride": [4, 4, 2],
    "padding": [10, 5, 5],
    "ch_nums": [1, 32, 64, 128],
    "act_fn": "ReLU",  # available options: ReLU, LeakyReLU, ELU, Tanh
    "act_fn_out": "Linear",
    "batch_normalization": "True",

    #######################
    # WavenetAE Params    #
    #######################
    "wavenet_input_channels": 1,
    "wavenet_hidden_channels": 32,
    "wavenet_bottleneck_channels": 16,
    "wavenet_encoder_blocks": 4,
    "wavenet_decoder_blocks": 4,
    "wavenet_kernel_size": 5,
    "wavenet_dilation_base": 2,
    "wavenet_dropout": 0.2,

    #######################
    # AttentionAE Params  #
    #######################
    "attention_input_channels": 1,
    "attention_hidden_channels": [32, 64, 128],  # Channels for each encoder layer
    "attention_kernel_sizes": [9, 5, 3],         # Kernel sizes for each layer
    "attention_strides": [4, 2, 2],              # Strides for each layer
    "attention_paddings": [4, 2, 1],             # Paddings for each layer
    "attention_dropout": 0.2,                    # Dropout rate
    "attention_head_dim": 64,                    # Dimension of each attention head
    "attention_num_heads": 4,                    # Number of attention heads
    "use_skip_connections": True,                # Whether to use skip connections

    #######################
    # Training Parameters #
    #######################
    "optimizer_name": "adam",
    "scheduler_name": "plateau",
    "loss_fn": "kl_divergence",
    "loss_domain": "frequency",
    "max_epoch": 5,
    "lr": 0.00001,
    "weight_decay": 0.01,
    "early_stop_monitor": "val_loss",
    "early_stop_mindelta": 0.00,
    "early_stop_patience": 40,
    "lr_monitor": "val_loss",
    "lr_decay_factor": 0.9,
    "lr_patience": 20,
    
    #######################
    # Anomaly Detection   #
    #######################
    "output_dir": "outputs",       # Base directory for all output files
    "plot_anomaly_scores": True,   # Enable anomaly score plotting
    "anomaly_threshold": None,     # Set to a float value to define detection threshold (e.g., 0.5)
    "clean_checkpoints": True,     # Whether to clean checkpoint directory before training
    
    # Plot customization options
    "anomaly_plot_config": {
        "figsize": (12, 6),       # Figure size in inches (width, height)
        "color": "blue",          # Line color for anomaly scores
        "threshold_color": "red", # Color for threshold line
        "grid": True,             # Show grid lines
        "dpi": 300                # Resolution of saved figure
    }
}