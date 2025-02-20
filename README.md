# Autoencoder-based Anomaly Detection
A deep learning framework for unsupervised anomaly detection in time series data using autoencoder architectures.

# Overview

This repository implements the unsupervised anomaly detection framework presented in:

> "Unsupervised anomaly detection of permanent-magnet offshore wind generators through electrical and electromagnetic measurements"  
> Wind Energy Science, 2024  
> DOI: 10.5194/wes-9-2063-2024

## Original Implementation
The initial implementation uses a convolutional autoencoder (CAE) model, as shown in Fig. 1, trained on electrical and electromagnetic time series data for anomaly detection in permanent-magnet wind turbine generators. The model is trained on normal operational data and evaluated on various fault conditions. For detailed methodology and results, please refer to the paper (you are welcome to cite if you find it interesting).

**Note**: The dataset used in the original work is private but can be made available upon request.

## Extended Framework
This repository extends the original work by implementing and evaluating additional autoencoder architectures and loss functions for time series anomaly detection.

### Implemented Architectures
- Convolutional Autoencoder (Original paper)
- Wavenet-based Autoencoder
- Attention-based Autoencoder

### Loss Functions
[To be added]
