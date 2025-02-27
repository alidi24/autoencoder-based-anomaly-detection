# Autoencoder-based Anomaly Detection
A deep learning framework for unsupervised anomaly detection in time series data using autoencoder architectures.

# Overview

This repository implements the unsupervised anomaly detection framework presented in:

> "Unsupervised anomaly detection of permanent-magnet offshore wind generators through electrical and electromagnetic measurements"
> Ali Dibaj, Mostafa Valavi, and Amir R. Nejad    
> Wind Energy Science, 2024  
> DOI: [https://doi.org/10.5194/wes-9-2063-2024](https://doi.org/10.5194/wes-9-2063-2024)

## Original Implementation
The initial implementation uses a convolutional autoencoder (CAE) model, as shown in Fig. 1, trained on electrical and electromagnetic time series data for anomaly detection in wind turbine permanent-magnet generators. The model is trained on normal operational data and evaluated on various fault conditions. For detailed methodology and results, please refer to the paper (we appreciate citations if you find this work useful for your research).

<p align="center">
  <img src="cae-model.png" alt="CAE model architecture" width="80%">
  <br>
  <em>Fig. 1: Overview of data processing and CAE model architecture implemented in the paper</em>
</p>

## Dataset
### Original Dataset
The dataset used in the original paper was generated using proprietary simulation software for wind turbine permanent-magnet generators and is confidential. It contains electrical and electromagnetic signals under both normal and various fault conditions.

### Repository Implementation 
For demonstration and validation purposes, this repository uses the Case Western Reserve University (CWRU) Bearing Vibration Dataset, which is publicly available. The CWRU dataset contains vibration measurements from normal and faulty bearings with different fault types and severities.

Dataset Source: [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)  
Hugging Face Version: A processed version is available via Hugging Face Datasets as [alidi/cwru-dataset](https://huggingface.co/datasets/alidi/cwru-dataset)

> **Note:** To access the dataset, you'll need a Hugging Face account and access token:
> ```python
> from huggingface_hub import login
> login(token="your_huggingface_token")
> ```
> To get your token: Sign up at huggingface.co → Settings → Access Tokens → Create new token

The data pipeline is designed to be modular, allowing users to easily adapt the framework to work with their own time series datasets.


## Extended Framework
This repository extends the original work by implementing and evaluating additional autoencoder architectures and loss functions for time series anomaly detection.

### Implemented Architectures
- Convolutional Autoencoder (Original paper)
- Wavenet-based Autoencoder
- Attention-based Autoencoder

### Loss Functions
The framework supports multiple loss functions that can be applied in both time and frequency domains:

- MSE, MAE, Huber, Cosine Similarity, KL Divergence

## Anomaly Detection Results

Below are the anomaly detection results for the three implemented models when using time-domain MSE as the loss function. Each plot shows the reconstruction error (anomaly score), with normal data (samples 0-1600) and faulty data (samples 1600-5200) from the CWRU bearing dataset - both drive-end and fan-end measurements and different load conditions.

<p align="center">
  <img src="outputs/anomaly_plots/CAE_mse_time_anomaly_scores.png" alt="CAE Anomaly Scores" width="80%">
  <br>
  <em>Fig. 2: Convolutional Autoencoder (CAE) anomaly scores with time-domain MSE loss</em>
</p>
<p align="center">
  <img src="outputs/anomaly_plots/WavenetAE_mse_time_anomaly_scores.png" alt="WaveNet Autoencoder Anomaly Scores" width="80%">
  <br>
  <em>Fig. 3: WaveNet Autoencoder anomaly scores with time-domain MSE loss</em>
</p>
<p align="center">
  <img src="outputs/anomaly_plots/AttentionAE_mse_time_anomaly_scores.png" alt="Attention Autoencoder Anomaly Scores" width="80%">
  <br>
  <em>Fig. 4: Attention Autoencoder anomaly scores with time-domain MSE loss</em>
</p>

The plots show how effectively each model distinguishes between normal and faulty bearing conditions. The Attention Autoencoder appears to be the most sensitive to anomalies, likely due to its ability to capture long-range dependencies in the signal.

## Requirements & Installation
This project uses Poetry for dependency management. The main dependencies are:
- Python >= 3.10
- PyTorch 2.2+
- Lightning 2.1+
- NumPy
- Librosa (for signal processing)
- Matplotlib (for visualization)
- Hugging Face Datasets

### Installation

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/alidibaj/autoencoder-based-anomaly-detection
cd autoencoder-based-anomaly-detection

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry env activate
```

### Training a Model
```bash
python train.py
```

### Model Configuration
You can customize the model architecture, loss function, training parameters, and more by modifying the configuration file [config.py](config.py)

```bash
config["which_model"] = "CAE"  # Options: CAE, WavenetAE, AttentionAE
config["loss_fn"] = "mse"  # Options: mse, mae, huber, cosine, kl_divergence, shape_factor, combined
config["loss_domain"] = "frequency"  # Options: time, frequency
```

## Project Structure

- `models/` - Autoencoder model definitions
- `src/` - Core functionality including data processing and training
- `train.py` - Main training script
- `config.py` - Configuration parameters

## License 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

