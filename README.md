# BrainTrain

Deep learning pipeline for brain MRI classification with explainability.

## Overview

Train and evaluate deep learning models on brain MRI data with explainability methods to visualize model predictions.

## Features

- **Training** - Train deep learning models on brain MRI scans
- **Testing** - Evaluate model performance on test sets
- **Explainability** - Generate GradCAM and saliency maps for predictions

## Project Structure

```
.
├── architectures/          # Neural network models
├── dataloaders/           # Dataset loaders
├── src/          # Neural network models
├── dataloaders/           # Dataset loaders
├── train.py               # Model training script
├── test.py                # Model evaluation script
├── heatmap.py             # GradCAM and saliency visualization
└── config.py              # Configuration and paths
```

## Quick Start

### 1. Train Model

```bash
python train.py
```

Trains the model on your brain MRI dataset and saves checkpoints.

### 2. Test Model

```bash
python test.py
```

Evaluates the trained model on the test set and reports performance metrics.

### 3. Generate Explainability Maps

```bash
python heatmap.py
```

## Configuration

Edit `config.py` to customize:
- Data paths
- Model architecture
- Training hyperparameters
- Explainability settings
