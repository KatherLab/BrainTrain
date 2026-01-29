# Bias Correction Methodology for Age Predictions

## Overview

This document describes the linear regression-based bias correction approach implemented for age predictions to reduce systematic bias and improve prediction accuracy.

## Methodology

### The Problem
Age prediction models may exhibit systematic bias (over/underestimation) that isn't removed by standard training. A linear bias correction can remove this systematic shift.

### The Solution
A two-stage approach to prevent **information leakage**:

1. **Stage 1 (Training Phase): Fit correction model on VALIDATION data**
2. **Stage 2 (Testing Phase): Apply correction coefficients to TEST data**

This ensures test set integrity and prevents circular dependencies.

## Implementation Details

### Correction Formula

The bias correction model fits a linear relationship:
```
true_value = β₀ + β₁ × predicted_value
```

Where:
- β₀ (intercept) = bias offset
- β₁ (slope) = scaling factor

Corrected predictions are then computed as:
```
corrected_prediction = (predicted_value - β₀) / β₁
```

### Stage 1: Fitting (During Training)

**When:** At the end of training, after best model is selected
**Where:** On validation set only
**Code location:** `train.py` (after validation epoch)

```python
from test import fit_and_save_bias_correction_coefficients

# After completing training and selecting best model
fit_and_save_bias_correction_coefficients(
    y_val_true=validation_labels,      # True validation labels
    y_val_pred=validation_predictions,  # Model predictions on validation set
    save_path=cfg.BIAS_CORRECTION_COEFFICIENTS_PATH
)
```

**Output:** JSON file with coefficients
```json
{
    "intercept": -2.5432,
    "slope": 1.0234,
    "r2_validation": 0.8543,
    "n_samples_validation": 500,
    "description": "Bias correction coefficients fitted on validation set...",
    "methodology": "Linear regression: true_value = intercept + slope * predicted_value"
}
```

### Stage 2: Application (During Testing)

**When:** During test.py execution
**Where:** Loads pre-fitted coefficients and applies to test predictions
**Code location:** `test.py` → `test_regression_metrics()` function

```python
# Loads coefficients from cfg.BIAS_CORRECTION_COEFFICIENTS_PATH
bias_correction_coeffs = load_bias_correction_coefficients(
    cfg.BIAS_CORRECTION_COEFFICIENTS_PATH
)

# Apply to test predictions
y_pred_corrected, correction_info = apply_linear_bias_correction(
    y_true=test_labels,
    y_pred=test_predictions,
    coefficients=bias_correction_coeffs,  # Pre-fitted on validation
    fit_on_data=False  # Never fit on test data!
)
```

**Outputs:**
- Both raw and corrected predictions saved to CSV
- Side-by-side comparison plots
- Metrics for both raw and corrected predictions
- Correction information stored in summary CSV

## Information Leakage Prevention

### ✅ CORRECT Approach (Implemented)
```
Training Phase:
├─ Train model on train set
├─ Validate on validation set
├─ Fit correction model on validation set predictions
└─ Save correction coefficients to file

Testing Phase:
├─ Load correction coefficients (from validation)
├─ Get test predictions
├─ Apply pre-fitted coefficients to test predictions
└─ Evaluate on corrected test predictions
     ↑ NO information leakage - test set never used to fit model
```

### ❌ WRONG Approach (Avoided)
```
Testing Phase:
├─ Get test predictions
├─ Fit correction model on test predictions vs test labels
├─ Apply fitted coefficients to same test predictions
└─ Evaluate
     ↑ INFORMATION LEAKAGE - test data used to fit correction model!
     ↑ Circular dependency - overly optimistic metrics!
```

## Configuration

In `config.py`:

```python
# Enable/disable bias correction
APPLY_BIAS_CORRECTION = True

# Path to save/load coefficients
BIAS_CORRECTION_COEFFICIENTS_PATH = f'{MODEL_DIR}/{TRAINING_MODE}/bias_correction_{COLUMN_NAME}.json'
```

## Output Files

During testing, the following files are generated:

1. **Predictions CSV**
   - `predicted_raw`: Original model predictions
   - `predicted_corrected`: Bias-corrected predictions

2. **Plots**
   - `scatter_raw.png`: Raw predictions vs actual
   - `scatter_corrected.png`: Corrected predictions vs actual
   - `bias_correction_comparison.png`: Side-by-side comparison
   - `residuals.png`: Residuals analysis (on corrected predictions)

3. **Metrics CSV**
   - Raw metrics (before correction)
   - Corrected metrics (after correction)
   - Bias correction coefficients and R²

## Reporting Metrics

In manuscripts and reports, clearly state:

**Template language:**
> "A linear regression-based bias correction was applied to reduce systematic bias in age predictions. 
> The correction model (β₀, β₁) was fitted on the validation set during the training phase and then 
> applied to the test set predictions, preventing information leakage. The resulting bias-corrected 
> predictions showed [X% improvement in MAE / r increased from A to B] compared to raw predictions."

## Verification Checklist

- [ ] Coefficients are fitted on **validation set only**
- [ ] Coefficients are saved to file before testing
- [ ] Test.py loads pre-fitted coefficients from file
- [ ] `fit_on_data=False` is used in `apply_linear_bias_correction()`
- [ ] Both raw and corrected metrics are reported
- [ ] Correction formulas and R² are documented in output files
- [ ] No warning about "fitting on test data" appears in console output

## References

This methodology follows best practices in machine learning to prevent information leakage and ensure valid evaluation of model performance improvements.

See: `fit_and_save_bias_correction_coefficients()` in `test.py` for training phase usage
See: `apply_linear_bias_correction()` in `test.py` for testing phase usage
