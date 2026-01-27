# Reproducibility Guide

**Operator-Based Heart-Brain Monitoring Framework**  
**Version 1.0**  
**Last Updated: January 27, 2026**

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Reproducing Paper Results](#reproducing-paper-results)
6. [Validation Checklist](#validation-checklist)
7. [Expected Outputs](#expected-outputs)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)

---

## Overview

This guide provides step-by-step instructions to reproduce all results presented in the manuscript:

> **"Operator-Based Heart–Brain Isostasis Monitoring via Triadic Spiral-Time Embeddings:  
> A Conservative, Falsifiable Framework for Early-Warning Detection in Coupled EEG/ECG Dynamics"**  
> Marcel Krüger and Don Feeney, 2026

### Reproducibility Goals

✓ **Deterministic results:** All random seeds fixed  
✓ **Transparent parameters:** All thresholds and weights preregistered  
✓ **Open datasets:** Publicly available or synthetic with ground truth  
✓ **Documented environment:** Exact software versions specified  
✓ **Automated pipelines:** Single-command execution where possible  

---

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 4 cores, 2.0 GHz
- RAM: 8 GB
- Storage: 10 GB free space
- OS: Linux, macOS, or Windows 10+

**Recommended:**
- CPU: 8+ cores, 3.0+ GHz
- RAM: 16+ GB
- Storage: 50+ GB (for large datasets)
- OS: Ubuntu 20.04+ or macOS 12+

### Software Requirements

**Core Dependencies:**
```
Python 3.8+
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
pandas >= 1.3.0
jupyter >= 1.0.0
```

**Optional Dependencies (for extended functionality):**
```
mne >= 0.24.0          # For EEG data loading (CHB-MIT format)
scikit-learn >= 1.0.0  # For additional metrics
seaborn >= 0.11.0      # For enhanced visualizations
```

**Development Tools (optional):**
```
pytest >= 6.2.0        # For running test suite
black >= 21.0          # For code formatting
pylint >= 2.10.0       # For code quality checks
```

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/dfeen87/Triadic-Biosignal-Monitor.git
cd Triadic-Biosignal-Monitor
```

### Step 2: Create Virtual Environment

**Using venv (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n heartbrain python=3.9
conda activate heartbrain
```

### Step 3: Install Dependencies

**Core installation:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Development installation (includes testing tools):**
```bash
pip install -r requirements-dev.txt
```

### Step 4: Verify Installation
```bash
python -c "import numpy, scipy, matplotlib; print('Installation successful!')"
```

**Run test suite (if pytest installed):**
```bash
pytest tests/ -v
```

---

## Dataset Preparation

### Option 1: Synthetic Datasets (Recommended for Quick Start)

**No download required.** Synthetic datasets are generated on-the-fly by the notebooks.

**Advantages:**
- Immediate execution
- Ground truth available
- Controlled noise and transition parameters
- No data privacy concerns

**To use:**
- Open any notebook (`notebooks/01_*.ipynb` through `06_*.ipynb`)
- Run all cells - synthetic data will be generated automatically

### Option 2: CHB-MIT Scalp EEG Database (For Real-World Validation)

**Source:** PhysioNet - [https://physionet.org/content/chbmit/1.0.0/](https://physionet.org/content/chbmit/1.0.0/)

**Download instructions:**
```bash
# Create datasets directory
mkdir -p datasets/chb-mit

# Download using wget (example for subject chb01)
wget -r -N -c -np -P datasets/chb-mit \
  https://physionet.org/files/chbmit/1.0.0/chb01/

# Or download manually from PhysioNet website
```

**Dataset structure:**
```
datasets/
└── chb-mit/
    ├── chb01/
    │   ├── chb01_01.edf
    │   ├── chb01_02.edf
    │   └── ...
    ├── chb02/
    └── ...
```

**Citation (required if using CHB-MIT):**
```
Shoeb, A. (2009). CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet.
https://doi.org/10.13026/C2K01R
```

### Option 3: Custom Datasets

**Format requirements:**

- **EEG:** `.edf`, `.fif`, or `.npy` format
- **ECG/HRV:** `.edf`, `.csv`, or `.npy` format
- **Sampling rate:** 250 Hz (recommended) or specify in config
- **Annotations:** Event times in seconds (for validation)

**Directory structure:**
```
datasets/
└── custom/
    ├── subject_001/
    │   ├── eeg.npy      # Shape: (n_samples,) or (n_channels, n_samples)
    │   ├── ecg.npy      # Shape: (n_samples,)
    │   └── events.csv   # Columns: 'time', 'type', 'description'
    └── subject_002/
        └── ...
```

---

## Reproducing Paper Results

### Quick Start: Run All Notebooks

**Execute all validation notebooks sequentially:**
```bash
# Convert notebooks to Python scripts and run
jupyter nbconvert --to script notebooks/*.ipynb
for notebook in notebooks/*.py; do
    python "$notebook"
done
```

**Or run interactively in Jupyter:**
```bash
jupyter notebook notebooks/
```

### Detailed Reproduction Steps

#### Result 1: Phase Extraction Demo (Figure 1, Manuscript)

**Notebook:** `notebooks/01_phase_extraction_demo.ipynb`

**Expected output:**
- Triadic embedding visualization for synthetic signal
- Phase ϕ(t) and phase derivative χ(t) plots
- Regime transition detection demonstration

**Runtime:** ~2 minutes

**Command line (non-interactive):**
```bash
jupyter nbconvert --to notebook --execute \
  --output 01_phase_extraction_demo_output.ipynb \
  notebooks/01_phase_extraction_demo.ipynb
```

**Verification:**
- Check that χ(t) shows clear regime change at t=30s
- Confirm phase unwrapping is continuous (no 2π jumps in plots)

---

#### Result 2: Feature Computation (Table 1, Manuscript)

**Notebook:** `notebooks/02_feature_computation.ipynb`

**Expected output:**
- ΔS, ΔI, ΔC computation examples
- ICE stability triangle visualization
- Feature comparison across synthetic regimes

**Runtime:** ~3 minutes

**Verification:**
- ΔS deviation should correlate with frequency changes
- ΔI deviation should correlate with entropy changes
- ΔC deviation should correlate with coupling strength changes

---

#### Result 3: EEG Validation (Figure 2, Tables 2-3, Manuscript)

**Notebook:** `notebooks/03_eeg_validation.ipynb`

**Expected output:**
- Lead-time distribution for preictal detection
- False alarm rate characterization
- ROC curves for detection performance

**Runtime:** ~10 minutes (synthetic), ~30 minutes (CHB-MIT if available)

**Key metrics to verify:**

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Lead time | 5-20 minutes | For preictal detection |
| False alarm rate | < 1 per hour | In interictal periods |
| AUC | > 0.85 | For regime detection |

**Command:**
```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 \
  --output 03_eeg_validation_output.ipynb \
  notebooks/03_eeg_validation.ipynb
```

---

#### Result 4: Synthetic Validation (Figure 3, Table 4, Manuscript)

**Notebook:** `notebooks/04_synthetic_validation.ipynb`

**Expected output:**
- Detection performance across regime types (frequency, amplitude, complexity)
- ROC curves with AUC > 0.90
- Noise sensitivity analysis

**Runtime:** ~15 minutes

**Verification criteria:**
```python
# All test cases should pass:
assert detection_rate['frequency_shift'] > 0.90
assert detection_rate['amplitude_transition'] > 0.85
assert detection_rate['complexity_transition'] > 0.80
assert mean_auc > 0.90
```

---

#### Result 5: Ablation Analysis (Figure 4, Tables 5-6, Manuscript)

**Notebook:** `notebooks/05_ablation_analysis.ipynb`

**Expected output:**
- Modality comparison (EEG-only vs ECG-only vs coupled)
- Feature component contributions (ΔS, ΔI, ΔC individually)
- Weight sensitivity analysis
- Threshold selection guidance

**Runtime:** ~20 minutes

**Key findings to verify:**

| Configuration | Expected Lead Time | Expected False Alarms |
|---------------|-------------------|-----------------------|
| EEG-only | 10-15 min | 2-3 per hour |
| ECG-only | 5-10 min | 3-5 per hour |
| Coupled | 15-20 min | 1-2 per hour |

**Interpretation:**
- Coupled mode should show superior performance
- ΔC (coupling) term should reduce false alarms
- Balanced weights (α=0.4, β=0.3, γ=0.3) should be robust

---

#### Result 6: Full Pipeline Demo (Figure 5, Clinical Report, Manuscript)

**Notebook:** `notebooks/06_full_pipeline_demo.ipynb`

**Expected output:**
- End-to-end processing from raw signals to clinical report
- Complete pipeline visualization (7-panel figure)
- Formatted clinical report with risk stratification
- Alert generation with lead-time analysis

**Runtime:** ~5 minutes

**Verification:**
- Pipeline completes without errors
- Clinical report generated with all sections
- Lead time is positive (early warning before event)
- All alerts have risk levels assigned

---

### Batch Reproduction Script

**Run all notebooks in sequence:**

Create `reproduce_all.sh`:
```bash
#!/bin/bash

set -e  # Exit on error

echo "====================================="
echo "Reproducing all paper results"
echo "====================================="

NOTEBOOKS=(
    "01_phase_extraction_demo"
    "02_feature_computation"
    "03_eeg_validation"
    "04_synthetic_validation"
    "05_ablation_analysis"
    "06_full_pipeline_demo"
)

for nb in "${NOTEBOOKS[@]}"; do
    echo ""
    echo "Running: $nb"
    echo "-------------------------------------"
    
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=3600 \
        --output "${nb}_output.ipynb" \
        "notebooks/${nb}.ipynb"
    
    echo "✓ Completed: $nb"
done

echo ""
echo "====================================="
echo "All results reproduced successfully!"
echo "====================================="
echo ""
echo "Output notebooks saved with '_output' suffix"
echo "Review notebooks/XX_*_output.ipynb for results"
```

**Run:**
```bash
chmod +x reproduce_all.sh
./reproduce_all.sh
```

---

## Validation Checklist

Use this checklist to verify complete reproducibility:

### Environment Setup

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Installation verified (`python -c "import numpy, scipy, matplotlib"`)
- [ ] Test suite passes (if applicable: `pytest tests/`)

### Notebook Execution

- [ ] `01_phase_extraction_demo.ipynb` runs without errors
- [ ] `02_feature_computation.ipynb` runs without errors
- [ ] `03_eeg_validation.ipynb` runs without errors
- [ ] `04_synthetic_validation.ipynb` runs without errors
- [ ] `05_ablation_analysis.ipynb` runs without errors
- [ ] `06_full_pipeline_demo.ipynb` runs without errors

### Result Verification

- [ ] Phase extraction shows clear regime transitions
- [ ] Feature computation produces expected deviation ranges
- [ ] EEG validation achieves lead time > 5 minutes
- [ ] Synthetic validation AUC > 0.90
- [ ] Ablation shows coupled mode superiority
- [ ] Full pipeline generates complete clinical report

### Output Files

- [ ] All output notebooks saved with `_output` suffix
- [ ] Figures match manuscript figures (visual inspection)
- [ ] Numerical results within 5% of published values
- [ ] No errors or warnings in notebook outputs

### Documentation

- [ ] README.md reviewed
- [ ] SCOPE_AND_SAFETY.md reviewed
- [ ] REPRODUCIBILITY.md (this document) followed
- [ ] VALIDATION_PROTOCOL.md reviewed (if planning prospective study)

---

## Expected Outputs

### Numerical Results Summary

**From Notebook 03 (EEG Validation):**
```
Expected Output:
  Lead time: 10-20 minutes
  False alarm rate: < 1 per hour
  Detection rate: > 85%
  AUC: > 0.85
```

**From Notebook 04 (Synthetic Validation):**
```
Expected Output:
  Frequency shift detection: > 90%
  Amplitude transition detection: > 85%
  Complexity transition detection: > 80%
  Mean AUC: > 0.90
  Noise robustness: Degrades gracefully, > 80% up to σ=0.5
```

**From Notebook 05 (Ablation Analysis):**
```
Expected Output:
  EEG-only lead time: 10-15 min
  ECG-only lead time: 5-10 min
  Coupled lead time: 15-20 min (best)
  Coupled false alarms: 1-2 per hour (lowest)
```

### Visual Outputs

**Expected figures (all notebooks):**

1. **Triadic embedding plots** showing (t, ϕ, χ)
2. **ICE triangle** with deviation trajectories
3. **Time series plots** with ground truth and detections
4. **ROC curves** with AUC values
5. **Performance comparisons** (bar charts, line plots)
6. **Full pipeline visualization** (7-panel figure)

**Figure quality:**
- DPI: 300 (publication quality)
- Format: PNG or PDF
- Size: 14-16 inches width for multi-panel figures
- All text legible at publication size

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'scipy'
```

**Solution:**
```bash
pip install --upgrade scipy numpy matplotlib pandas
```

---

#### Issue 2: Jupyter Kernel Not Found

**Error:**
```
Kernel not found for jupyter notebook
```

**Solution:**
```bash
python -m ipykernel install --user --name=venv
```

---

#### Issue 3: Memory Error on Large Datasets

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
- Reduce window size in config
- Process shorter signal segments
- Increase system RAM or use smaller dataset

**Code modification:**
```python
# In notebook, reduce memory usage:
config.window_size = 10  # Reduce from 15
config.step_size = 5     # Reduce from 3
```

---

#### Issue 4: Results Don't Match Paper Exactly

**Possible causes:**
1. **Random seed not set:** Check that `np.random.seed(42)` is present
2. **Different library versions:** Verify numpy/scipy versions
3. **Floating point precision:** Small differences (<1%) are acceptable
4. **Dataset differences:** Ensure using same dataset/subject

**Verification:**
```python
# Check versions:
import numpy, scipy, matplotlib
print(f"numpy: {numpy.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"matplotlib: {matplotlib.__version__}")

# Should output:
# numpy: 1.21.x or higher
# scipy: 1.7.x or higher
# matplotlib: 3.4.x or higher
```

---

#### Issue 5: Notebook Execution Timeout

**Error:**
```
Timeout waiting for execute reply (30s).
```

**Solution:**
```bash
# Increase timeout:
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/XX_notebook.ipynb
```

---

#### Issue 6: Figure Display Issues

**Problem:** Figures not displaying in notebook

**Solution:**
```python
# Add to top of notebook:
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
```

---

### Getting Help

**If you encounter issues not covered here:**

1. **Check existing issues:** [GitHub Issues](https://github.com/dfeen87/Triadic-Biosignal-Monitor/issues)
2. **Search discussions:** [GitHub Discussions](https://github.com/dfeen87/Triadic-Biosignal-Monitor/discussions)
3. **Open new issue:** Provide:
   - Full error message
   - System information (`python --version`, OS)
   - Steps to reproduce
   - Expected vs. actual behavior

**Issue template:**
```markdown
**Describe the bug:**
[Clear description]

**To reproduce:**
1. Run notebook: XX_notebook.ipynb
2. Execute cell: [cell number]
3. See error: [error message]

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- numpy: [version]
- scipy: [version]

**Expected behavior:**
[What should happen]

**Actual behavior:**
[What actually happened]
```

---

## Contributing

### How to Contribute

We welcome contributions to improve reproducibility!

**Areas where contributions are helpful:**
- Bug fixes in notebooks
- Documentation improvements
- Additional dataset examples
- Performance optimizations
- Extended validation on new datasets

### Contribution Workflow

1. **Fork repository**
2. **Create feature branch:** `git checkout -b feature/my-improvement`
3. **Make changes** with clear documentation
4. **Test thoroughly** on clean environment
5. **Submit pull request** with description of changes

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Include comments for complex operations
- Run `black` formatter before committing (if available)

### Testing Contributions

**Before submitting:**

- [ ] All notebooks execute without errors
- [ ] Results remain reproducible
- [ ] Documentation updated
- [ ] No breaking changes to API (or documented if necessary)
}

**Additionally cite the signal processing methods used:**

- Hilbert transform: Gabor, D. (1946). Theory of communication. *J. IEE*, 93, 429–457.
- Phase synchronization: Rosenblum et al. (1996). *Phys. Rev. Lett.*, 76, 1804.
- Heart rate variability: Task Force (1996). *Circulation*, 93, 1043–1065.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-27 | Initial reproducibility guide with complete instructions |

---

## Acknowledgments

- PhysioNet for CHB-MIT dataset
- scipy/numpy/matplotlib development teams
- Jupyter project for notebook infrastructure
- All contributors to open-source scientific Python ecosystem

---

**END OF REPRODUCIBILITY GUIDE**
