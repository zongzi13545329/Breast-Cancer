# EMBED Breast Cancer Recall Reduction with Racial Equity

## Project Overview

This codebase implements a longitudinal temporal AI model for reducing false-positive recalls in breast cancer screening, with explicit focus on racial equity using the EMBED dataset (42% African American).

### Key Innovations
1. **Longitudinal Temporal Comparison**: Siamese network comparing current vs. prior exams
2. **Algorithmic Fairness**: Race-conditional adaptation ensuring equity across demographics
3. **Clinical Impact**: 30-50% recall reduction while maintaining sensitivity

### Dataset
- **EMBED**: 116,000 patients, 365,000 screening exams (2013-2020)
- **Population**: 42% African American, 39% White, 7% Asian, 6% Hispanic
- **Task**: Predict unnecessary recalls (BI-RADS 0 → benign outcomes)

## Environment Setup

### Requirements
```bash
# Core dependencies
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.7 (recommended)

# Main packages
torch, torchvision
numpy, pandas, scikit-learn
matplotlib, seaborn
opencv-python, pillow
pydicom (for DICOM handling)
monai (medical imaging)
timm (vision transformers)
tensorboard
wandb (optional, for experiment tracking)
```

### Installation
```bash
# Clone repository
git clone <repo-url>
cd embed_recall_reduction

# Create conda environment
conda create -n embed python=3.9
conda activate embed

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
embed_recall_reduction/
├── configs/                    # Configuration files
│   ├── base_config.yaml
│   ├── temporal_model.yaml
│   └── fairness_model.yaml
├── data/                       # Data processing
│   ├── dataset.py             # Dataset classes
│   ├── preprocessing.py       # DICOM processing, augmentation
│   ├── dataloader.py          # Data loaders
│   └── stratified_split.py    # Race-stratified splits
├── models/                     # Model architectures
│   ├── backbone.py            # ResNet, ViT encoders
│   ├── temporal_model.py      # Siamese temporal network
│   ├── fairness_model.py      # Race-conditional adapters
│   ├── multitask_model.py     # Multi-task learning (optional)
│   └── losses.py              # Custom loss functions
├── training/                   # Training scripts
│   ├── trainer.py             # Main training loop
│   ├── validator.py           # Validation logic
│   └── callbacks.py           # Training callbacks
├── evaluation/                 # Evaluation scripts
│   ├── metrics.py             # Performance metrics
│   ├── fairness_metrics.py    # Fairness evaluation
│   ├── statistical_tests.py   # DeLong test, bootstrap CI
│   └── visualizations.py      # ROC curves, calibration plots
├── utils/                      # Utilities
│   ├── logger.py              # Logging utilities
│   ├── checkpoint.py          # Model checkpointing
│   └── config_parser.py       # Config handling
├── scripts/                    # Executable scripts
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── inference.py           # Inference script
│   └── analyze_fairness.py    # Fairness analysis
├── notebooks/                  # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── baseline_analysis.ipynb
│   └── results_visualization.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Data Preparation
```bash
# Preprocess EMBED DICOM files
python scripts/preprocess_data.py \
    --input_dir /path/to/embed/dicoms \
    --output_dir ./data/processed \
    --image_size 1024
```

### 2. Training Baseline Model
```bash
# Single-image baseline
python scripts/train.py \
    --config configs/baseline.yaml \
    --data_dir ./data/processed \
    --output_dir ./experiments/baseline
```

### 3. Training Temporal Model
```bash
# Longitudinal temporal model
python scripts/train.py \
    --config configs/temporal_model.yaml \
    --data_dir ./data/processed \
    --output_dir ./experiments/temporal
```

### 4. Training Fairness Model
```bash
# Race-conditional fairness model
python scripts/train.py \
    --config configs/fairness_model.yaml \
    --data_dir ./data/processed \
    --output_dir ./experiments/fairness \
    --use_race_conditioning
```

### 5. Evaluation
```bash
# Comprehensive evaluation with fairness metrics
python scripts/evaluate.py \
    --checkpoint ./experiments/fairness/best_model.pth \
    --data_dir ./data/processed \
    --output_dir ./results \
    --analyze_fairness
```

## Key Features

### 1. Temporal Architecture
- Siamese network with shared weights
- Spatial Transformer Network for automatic alignment
- Cross-attention for temporal correspondence
- Change detection module

### 2. Fairness Components
- Race-conditional adapters (5% parameters per race)
- Adversarial debiasing (optional)
- Fairness constraint losses (Demographic Parity, Equalized Odds)
- Comprehensive subgroup analysis

### 3. Training Features
- Mixed precision training (AMP)
- Gradient accumulation for large batch sizes
- Learning rate scheduling (cosine, step, plateau)
- Early stopping with patience
- Model checkpointing (best & last)
- TensorBoard logging
- WandB integration (optional)

### 4. Evaluation Metrics
- Standard: AUC, Sensitivity, Specificity, PPV, NPV
- Fairness: FPR/TPR parity, Equalized Odds, Calibration by race
- Statistical: DeLong test, Bootstrap CI (95%), Permutation tests
- Clinical: Decision curve analysis, Net benefit

## Expected Results

### Performance Targets
```
Baseline (Single Image):
- AUC: 0.85-0.87
- Specificity: 90%
- Recall Rate: 10%
- PPV: 4%

Temporal Model:
- AUC: 0.88-0.90
- Specificity: 93%
- Recall Rate: 7% (30% reduction)
- PPV: 5.7%

Fairness Model:
- Overall AUC: 0.88-0.90
- By Race AUC: All within 0.02 (no significant difference)
- FPR disparity: <2% (p > 0.05)
```

## Citation

If you use this codebase, please cite:
```
@article{embed_recall_reduction_2025,
  title={Improving Breast Cancer Screening Equity Through Longitudinal AI Analysis: A Multi-Ethnic Cohort Study},
  author={[Your Name] et al.},
  journal={[Target Journal]},
  year={2025}
}
```

## License

[Your License Choice]

## Contact

For questions or issues:
- Email: [Your Email]
- GitHub Issues: [Issue Tracker]

## Acknowledgments

- EMBED dataset provided by Emory University
- Research supported by [Funding Sources]
