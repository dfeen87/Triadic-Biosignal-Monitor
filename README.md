# Triadic-Biosignal-Monitor

**A Deterministic Software Framework for Early-Warning Detection in Coupled EEG–ECG Time Series**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Preprint-00CCBB.svg)](https://www.researchgate.net/publication/400100860_Operator-Based_Heart-Brain_Isostasis_Monitoring_via_Triadic_Spiral-Time_Embeddings_A_Conservative_Falsifiable_Framework_for_Early-Warning_Detection_in_Coupled_EEGECG_Dynamics)

## Overview

`Triadic-Biosignal-Monitor` is a research-grade signal analysis framework for detecting instability in coupled EEG and ECG/HRV time series. The system uses operator-based phase embeddings to provide interpretable early-warning detection **without machine learning black boxes**.

**Key Features:**
- 🔬 **Deterministic & Interpretable** - Every alert traceable to explicit signal features
- 🧪 **Ablation-Ready** - Built-in EEG-only, ECG-only, and coupled analysis modes
- ⚙️ **Fixed Thresholds** - Preregistered parameters for prospective validation
- 📊 **Full Explainability** - Component-level logging and confidence scoring
- 🛡️ **Conservative Design** - Fail-safe behavior and artifact handling

### Scope Statement

This software is designed for **monitoring and decision support only**. It does not:
- ❌ Perform therapeutic stimulation or closed-loop actuation
- ❌ Claim new physical or physiological laws
- ❌ Provide clinical diagnosis or treatment recommendations

All clinical utility requires prospective validation and appropriate regulatory oversight.

---

## Theoretical Foundation

The framework implements the methodology described in:

- **Krüger & Feeney (2026)**: "Operator-Based Heart–Brain Isostasis Monitoring via Triadic Spiral-Time Embeddings"
- **Krüger & Feeney (2026)**: "A Deterministic Software Framework for Early-Warning Detection in Coupled EEG–ECG Time Series"

### Core Mathematics

**Triadic Phase Embedding:**
```
ψ(t) = (t, ϕ(t), χ(t))
```
where `ϕ(t)` is instantaneous phase and `χ(t) = ∂ϕ/∂t` is phase torsion.

**Instability Functional:**
```
ΔΦ(t) = α|ΔS(t)| + β|ΔI(t)| + γ|ΔC(t)|
```
- **ΔS**: Spectral/morphological deviation
- **ΔI**: Information/entropy deviation  
- **ΔC**: EEG–ECG coupling deviation

**Decision Gate:**
```
G(t) = 1{ΔΦ(t) ≥ τ}
```

---

## Installation

### Requirements
- Python 3.8+
- NumPy, SciPy
- MNE (EEG processing)
- NeuroKit2 (ECG/HRV analysis)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/dfeen87/triadic-biosignal-monitor.git
cd triadic-biosignal-monitor

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Try a demo notebook
jupyter notebook notebooks/01_phase_extraction_demo.ipynb
```

---

## Repository Structure

```
triadic-biosignal-monitor/
├── README.md                             # This file
├── LICENSE                               # MIT License
├── PAPER.md                              # Full paper in markdown format
├── requirements.txt                      # Python dependencies
├── setup.py                              # Package installation script
├── .gitignore                           # Git ignore patterns
│
├── core/                                 # Core signal processing modules
│   ├── __init__.py                      # Package initialization
│   ├── phase.py                         # Hilbert phase extraction, unwrap, robust derivative
│   ├── features.py                      # ΔS, ΔI, ΔC computation with baseline normalization
│   ├── gate.py                          # ΔΦ instability functional and decision gate
│   ├── metrics.py                       # Lead-time, false alarms/hr, ROC, ablations
│   └── preprocessing.py                 # Artifact rejection, filtering, synchronization
│
├── pipelines/                           # Analysis pipelines
│   ├── __init__.py
│   ├── eeg_only.py                      # EEG-only ablation pipeline
│   ├── ecg_only.py                      # ECG-only ablation pipeline
│   ├── coupled.py                       # Full coupled EEG-ECG pipeline
│   └── streaming.py                     # Near-real-time streaming mode (Phase 2)
│
├── datasets/                            # Dataset management
│   ├── __init__.py
│   ├── loaders.py                       # Dataset loading utilities
│   └── synthetic.py                     # Synthetic regime-change signal generators
│
├── configs/                             # Configuration files
│   ├── default.yaml                     # Default parameters (α, β, γ, τ)
│   ├── eeg_only.yaml                    # EEG-only ablation config
│   └── ecg_only.yaml                    # ECG-only ablation config
│
├── notebooks/                           # Jupyter notebooks for validation & demos
│   ├── 01_phase_extraction_demo.ipynb   # Demonstrates ϕ(t) and χ(t) extraction
│   ├── 02_feature_computation.ipynb     # Shows ΔS, ΔI, ΔC calculation
│   ├── 03_eeg_validation.ipynb          # Reproduce EEG dataset results from paper
│   ├── 04_synthetic_validation.ipynb    # Synthetic regime-change detection tests
│   ├── 05_ablation_analysis.ipynb       # Compare EEG-only vs ECG-only vs coupled
│   └── 06_full_pipeline_demo.ipynb      # End-to-end demonstration
│
├── docs/                                # Comprehensive documentation
│   ├── SCOPE_AND_SAFETY.md              # Medical-device safe language, limitations
│   ├── REPRODUCIBILITY.md               # Step-by-step reproduction of paper results
│   └── VALIDATION_PROTOCOL.md           # Prospective validation guidelines
│
├── tests/                               # Test suite
│   ├── __init__.py
│   ├── test_phase.py                    # Tests for phase.py
│   ├── test_features.py                 # Tests for features.py
│   ├── test_gate.py                     # Tests for gate.py
│   ├── test_metrics.py                  # Tests for metrics.py
│   ├── test_preprocessing.py            # Tests for preprocessing.py
│   └── test_pipelines.py                # Integration tests for pipelines
│
└── scripts/                             # Command-line utilities
    ├── run_validation.py                # Batch validation runner
    └── generate_report.py               # Performance report generator/
```

### Directory Descriptions

**Core Modules (`/core/`)**
- Production-ready signal processing implementations
- All functions deterministic with explicit error handling
- Comprehensive docstrings and type hints

**Pipelines (`/pipelines/`)**
- High-level analysis workflows
- Built-in ablation support for scientific validation
- Extensible base class for custom pipelines

**Datasets (`/datasets/`)**
- Standardized loaders for common EEG/ECG formats (EDF, FIF, WFDB)
- Synthetic data generators for testing
- Data quality validation utilities

**Configs (`/configs/`)**
- YAML-based configuration system
- All parameters preregistered and documented
- No runtime modification of thresholds

**Notebooks (`/notebooks/`)**
- Reproducibility-focused demonstrations
- Step-by-step validation of paper results
- Interactive exploration tools

**Documentation (`/docs/`)**
- Comprehensive guides for users and developers
- Safety and scope statements
- Regulatory compliance information

**Tests (`/tests/`)**
- >90% code coverage target
- Unit tests for all core functions
- Integration tests for full pipelines
- Performance regression tests

**Scripts (`/scripts/`)**
- Command-line tools for batch processing
- Report generation utilities
- Benchmarking and profiling tools

---

## Quick Usage

### Basic EEG-ECG Analysis

```python
from core.phase import triadic_embedding
from core.features import compute_delta_S, compute_delta_I, compute_delta_C
from core.gate import compute_instability_functional, decision_gate
import yaml

# Load configuration
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# Extract phase embeddings
psi_eeg = triadic_embedding(eeg_signal, fs=256)
psi_ecg = triadic_embedding(ecg_signal, fs=256)

# Compute deviations from baseline
delta_S = compute_delta_S(eeg_signal, baseline_eeg, fs=256, modality='eeg')
delta_I = compute_delta_I(eeg_signal, baseline_eeg, fs=256)
delta_C = compute_delta_C(eeg_signal, ecg_signal, baseline_coherence, fs=256)

# Evaluate instability functional
params = config['instability_functional']
delta_phi = compute_instability_functional(
    delta_S, delta_I, delta_C,
    alpha=params['alpha'],
    beta=params['beta'],
    gamma=params['gamma']
)

# Apply decision gate
alerts = decision_gate(delta_phi, threshold=params['threshold'])
```

### Running Ablation Pipelines

```python
from pipelines import eeg_only, ecg_only, coupled

# EEG-only analysis (no coupling term)
results_eeg = eeg_only.run_eeg_only_pipeline(eeg_signal, config)

# ECG-only analysis (no coupling term)
results_ecg = ecg_only.run_ecg_only_pipeline(ecg_signal, config)

# Full coupled analysis (all three terms)
results_coupled = coupled.run_coupled_pipeline(eeg_signal, ecg_signal, config)

# Compare ablations to prove coupling contribution
from core.metrics import ablation_comparison
comparison = ablation_comparison(results_eeg, results_ecg, results_coupled)
```

---

## Configuration

All parameters are specified in YAML configuration files with **fixed, preregistered values** for prospective validation.

### Example: `configs/default.yaml`

```yaml
instability_functional:
  alpha: 0.4        # Spectral/morphological weight
  beta: 0.3         # Information/entropy weight
  gamma: 0.3        # Coupling weight
  threshold: 2.5    # Decision gate threshold τ

baseline:
  window_seconds: 60
  method: 'median'

preprocessing:
  eeg:
    sampling_rate: 256
    bandpass: [0.5, 50]
  ecg:
    sampling_rate: 256
    bandpass: [0.5, 40]

features:
  delta_I:
    method: 'permutation_entropy'
  delta_C:
    method: 'PLV'  # Phase Locking Value
```

---

## Core Modules

### `core/phase.py` - Phase Extraction

**Key Functions:**
- `analytic_signal(signal, fs)` - Compute analytic signal via Hilbert transform
- `extract_phase(signal, fs, bandpass=None)` - Extract instantaneous phase ϕ(t)
- `unwrap_phase(phase)` - Remove 2π discontinuities
- `phase_derivative(phase, fs, method='savgol')` - Compute χ(t) = ∂ϕ/∂t
- `triadic_embedding(signal, fs)` - Return ψ(t) = (t, ϕ(t), χ(t))

### `core/features.py` - Feature Computation

**Key Functions:**
- `compute_delta_S(signal, baseline, fs, modality)` - Spectral/morphological deviation
  - EEG: bandpower drift (δ, θ, α, β, γ), spectral centroid
  - ECG/HRV: LF/HF ratio, RMSSD, SDNN, morphology
- `compute_delta_I(signal, baseline, fs, method)` - Information/entropy deviation
  - Methods: permutation entropy, sample entropy, Lempel-Ziv
- `compute_delta_C(signal_eeg, signal_ecg, baseline, fs)` - Coupling deviation
  - Methods: Phase Locking Value (PLV), magnitude-squared coherence (MSC)

### `core/gate.py` - Instability Detection

**Key Functions:**
- `compute_instability_functional(delta_S, delta_I, delta_C, alpha, beta, gamma)` - Compute ΔΦ(t)
- `decision_gate(delta_phi, threshold)` - Binary gate G(t) = 1{ΔΦ(t) ≥ τ}
- `generate_alert(gate_signal, timestamps, delta_phi, components)` - Explainable alerts

### `core/metrics.py` - Performance Evaluation

**Key Functions:**
- `compute_lead_time(predictions, ground_truth, timestamps)` - Lead-time distribution
- `false_alarm_rate(predictions, ground_truth, duration)` - False alarms per hour
- `roc_analysis(delta_phi, ground_truth, thresholds)` - ROC curve generation
- `ablation_comparison(results_eeg, results_ecg, results_coupled)` - Compare ablation modes

---

## Validation & Reproducibility

### Phase 1: EEG Validation (Current)

**Goal:** Reproduce EEG validation results from reference notebook

**Notebooks:**
- `03_eeg_validation.ipynb` - Reproduce paper figures
- `04_synthetic_validation.ipynb` - Synthetic regime-change detection

**Success Criteria:**
- ✅ Same features as paper
- ✅ Same thresholds
- ✅ Same lead-time plots

### Phase 2: Multi-Modal Validation

**Goal:** Demonstrate coupling contribution via ablations

**Deliverables:**
- ECG/HRV dataset integration
- False alarm rate analysis
- Robustness testing (noise, artifacts)

**Success Criteria:**
- ✅ False alarm rate < X per hour (preregistered)
- ✅ Coupling term shows added value
- ✅ Artifact handling robust

### Phase 3: Streaming & Optimization

**Goal:** Near-real-time processing capability

**Features:**
- Sliding-window processing
- Latency profiling
- Fail-safe behavior

---

## Safety & Design Principles

### Deterministic Behavior
- ✅ Fixed thresholds (no adaptive modification during deployment)
- ✅ Bounded operators only
- ✅ Preregistered parameters

### Explainability
- ✅ Every alert traceable to ΔΦ components
- ✅ Component-level logging (ΔS, ΔI, ΔC)
- ✅ Confidence flags for artifacts/missing data

### Fail-Safe Requirements
- ✅ Signal dropout → passive monitoring, no alerts
- ✅ Poor signal quality → "no-decision" state
- ✅ Computational error → fail-safe shutdown

### Conservative Design
- ✅ Explicit artifact handling
- ✅ Conservative thresholds
- ✅ No black-box decisions

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_phase.py
pytest tests/test_features.py
pytest tests/test_gate.py

# Run with coverage
pytest --cov=core tests/
```

**Test Coverage:**
- Unit tests for each core function
- Integration tests for full pipelines
- Validation tests reproducing paper results
- Performance tests (latency, memory)

---

## Documentation

Comprehensive documentation available in `/docs/`:

- **[SCOPE_AND_SAFETY.md](docs/SCOPE_AND_SAFETY.md)** - Medical-device safe language, regulatory guidance
- **[IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)** - Technical implementation details
- **[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** - How to reproduce paper results
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Function signatures and usage
- **[VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md)** - Prospective validation guidelines

---

## Dependencies

```
# Core numerical computing
numpy>=1.24.0
scipy>=1.10.0

# Signal processing
mne>=1.3.0              # EEG processing
neurokit2>=0.2.0        # ECG/HRV analysis
PyWavelets>=1.4.0       # Wavelet transforms

# Entropy and complexity
antropy>=0.1.6          # Entropy measures
nolds>=0.5.2            # Nonlinear dynamics

# Configuration
pyyaml>=6.0
h5py>=3.8.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.3.0
pytest-cov>=4.0.0

# Notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0
```

---

## Roadmap

### ✅ Phase 1: Reference Implementation
- [x] Core modules (phase, features, gate, metrics)
- [x] EEG-only pipeline
- [x] Validation notebooks
- [x] Documentation framework

### 🚧 Phase 2: Multi-Modal Validation (In Progress)
- [ ] ECG/HRV pipeline integration
- [ ] Full ablation analysis
- [ ] False alarm rate characterization
- [ ] Artifact robustness testing

### 📋 Phase 3: Streaming & Optimization (Planned)
- [ ] Near-real-time streaming pipeline
- [ ] Latency optimization
- [ ] Resource-aware processing
- [ ] Clinical deployment toolkit

---

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

**Priority Areas:**
- Additional entropy/complexity measures
- Enhanced artifact detection methods
- Performance optimizations
- Clinical dataset integrations

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{triadic_biosignal_monitor_2026,
  author = {Krüger, Marcel and Feeney, Don},
  title = {triadic-biosignal-monitor: A Deterministic Framework for 
           Early-Warning Detection in Coupled EEG–ECG Time Series},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/dfeen87/triadic-biosignal-monitor}
}

@article{kruger2026operator,
  author = {Krüger, Marcel and Feeney, Don},
  title = {Operator-Based Heart–Brain Isostasis Monitoring via 
           Triadic Spiral-Time Embeddings},
  year = {2026},
  note = {Preprint}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

**Marcel Krüger**  
Independent Researcher, Meiningen, Germany  
📧 marcelkrueger092@gmail.com  
🔗 ORCID: [0009-0002-5709-9729](https://orcid.org/0009-0002-5709-9729)

**Don Feeney**  
Independent Researcher, Pennsylvania, USA  
📧 dfeen87@gmail.com  
🔗 ORCID: [0009-0003-1350-4160](https://orcid.org/0009-0003-1350-4160)

---

## Acknowledgments

This work builds on established signal processing and time-series analysis methods. We acknowledge the open-source communities behind MNE, NeuroKit2, and SciPy.

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/dfeen87/triadic-biosignal-monitor/issues)
- **Discussions:** [GitHub Discussions](https://github.com/dfeen87/triadic-biosignal-monitor/discussions)
- **Email:** marcelkrueger092@gmail.com, dfeen87@gmail.com

---

**Status:** Active Development | Reference Implementation Phase  
**Last Updated:** January 27, 2026
