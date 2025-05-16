# EEG Analysis Baselines

This directory contains baseline implementations for various EEG analysis tasks, including case vs. control classification and treatment response prediction. The baselines include both deep learning and traditional machine learning approaches.

## Directory Structure

```
baselines_release/
├── biot/              # BIOT model implementation
├── cbramod/           # CBRAMOD model implementation
├── cnn/              # CNN-based models
├── handcrafted/      # Traditional ML approaches
├── labram/           # LABRAM model implementation
├── data_utils/       # Common data utilities
├── evaluate_case_control.py    # Evaluation script
└── eval_case_control.sh       # Evaluation shell script
```

## Available Baselines

### 1. CBRAMOD
A transformer-based model for EEG analysis with the following features:
- Pre-trained weights available
- Support for case vs. control classification
- Treatment response prediction (immediate and meaningful)
- Fine-tuning capabilities

### 2. BIOT
A specialized model for EEG analysis with:
- Custom architecture for EEG signals
- Support for multiple classification tasks
- Pre-trained model weights

### 3. CNN Models
Traditional CNN-based approaches for EEG analysis:
- Various CNN architectures
- Support for different input formats
- Standard deep learning pipeline

### 4. Handcrafted Features
Traditional machine learning approaches using:
- Feature engineering
- Classical ML algorithms
- Statistical analysis

### 5. LABRAM
A specialized model for EEG analysis with:
- Custom architecture
- Support for multiple tasks
- Pre-trained weights

## Usage

### Case vs. Control Classification

```bash
# Run CBRAMOD baseline
cd cbramod
./run_finetune_case_vs_control.sh

# Run evaluation
./inference_cbramod_case_vs_control.sh
```

### Treatment Response Prediction

```bash
# Immediate response prediction
cd cbramod
./run_finetune_immediate_responder.sh
./inference_cbramod_immediate_responder.sh

# Meaningful response prediction
./run_finetune_meaningful_responder.sh
./inference_cbramod_meaningful_responder.sh
```

### Evaluation

```bash
# Run evaluation script
./eval_case_control.sh
```

## Requirements

The project requires Python 3.x and the following key packages:
- PyTorch (2.2.1)
- PyTorch Lightning
- DeepSpeed
- MNE (1.4.2)
- Scikit-learn
- XGBoost
- Other dependencies listed in requirements.txt

Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Training

Each baseline implementation includes:
- Training scripts
- Configuration files
- Pre-trained weights (where applicable)
- Evaluation scripts

### Common Training Parameters
- Batch size: 64
- Learning rate: 3e-5
- Optimizer: AdamW
- Weight decay: 5e-2
- Dropout: 0.1

## Data Format
Please use the [IESSEEG-toolbox](https://github.com/roychowdhuryresearch/IESSEEG-toolbox) for preprocessing BIDS format data. The baselines expect post-preprocessed EEG data in the following format:
- Sampling frequency: 200Hz
- Preprocessed using scripts from preprocessing_release
- Metadata in CSV format
- Support for both sleep and awake recordings

## Notes

- All models support both sleep and awake EEG recordings
- Pre-trained weights are available for some models
- Evaluation metrics include accuracy, F1-score, and other relevant metrics
- Models can be fine-tuned on custom datasets
- Support for both binary and multi-class classification tasks 