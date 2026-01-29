# Automatic Bias Correction Calibration in test.py

## Overview

The test.py script now automatically calibrates bias correction on the validation set if pre-fitted coefficients don't exist. This provides a flexible workflow:

## How It Works

### Priority Order

When running `test.py`:

1. **Check for saved coefficients** → Use if found
2. **Check for validation CSV** → Calibrate on validation set if found
3. **Skip calibration** → If neither available

### Workflow

```
test.py execution:
├─ Load test set
├─ Load model
├─ Check if cfg.BIAS_CORRECTION_COEFFICIENTS_PATH exists
│
├─ IF file exists:
│  └─ Load pre-fitted coefficients → Use on test predictions
│
├─ IF file doesn't exist BUT validation set available:
│  ├─ Load validation set
│  ├─ Get validation predictions from model
│  ├─ Fit correction model on validation predictions vs labels
│  ├─ Save coefficients to file (for future use)
│  └─ Apply to test predictions
│
└─ IF neither available:
   └─ Skip bias correction, report metrics without correction
```

## Usage

### First Run (No Coefficients File)

```bash
python test.py -c age -m lora -g cuda:0
```

**Output:**
```
Calibrating bias correction using validation set...
Getting validation predictions for bias correction calibration...
Validation: 100%|████| 50/50 [00:30<00:00,  0.60s/it]
✓ Calibrated and saved to: ../models/lora/bias_correction_age.json
  Intercept: -2.5432
  Slope: 1.0234
  R² on validation: 0.8543
...
```

The JSON file is now saved and will be reused for future test runs.

### Subsequent Runs (Coefficients File Exists)

```bash
python test.py -c age -m lora -g cuda:0
```

**Output:**
```
✓ Loaded pre-fitted bias correction coefficients from:
  ../models/lora/bias_correction_age.json
  Intercept: -2.5432
  Slope: 1.0234
...
```

Much faster - loads from file directly.

## Key Points

✓ **Automatic:** No manual calibration needed  
✓ **Efficient:** Saves coefficients for reuse  
✓ **Safe:** Uses validation set, not test set  
✓ **Flexible:** Works with or without pre-fitted coefficients  
✓ **Documented:** Saves metadata in JSON file  

## Configuration

Required in `config.py`:

```python
# Paths
CSV_VAL = f'../data/{TRAIN_COHORT}/val/{CSV_NAME_TRAIN}.csv'
TENSOR_DIR = f'../images/{TRAIN_COHORT}/npy96'

# Settings
APPLY_BIAS_CORRECTION = True
BIAS_CORRECTION_COEFFICIENTS_PATH = f'{MODEL_DIR}/{TRAINING_MODE}/bias_correction_{COLUMN_NAME}.json'
```

## Output Files

Same as before:
- `{experiment}_scatter_raw.png` - Raw predictions
- `{experiment}_scatter_corrected.png` - Bias-corrected predictions
- `{experiment}_bias_correction_comparison.png` - Side-by-side comparison
- Summary CSV with both raw and corrected metrics
- Coefficients saved to JSON (reused on next run)

## Notes

- Validation set is loaded automatically if `cfg.CSV_VAL` exists
- Coefficients are derived from validation set, preventing test set leakage
- Same coefficients are used across multiple test runs (efficient)
- Can manually regenerate by deleting the JSON file and re-running test.py
