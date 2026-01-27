# Operator-Based Heart–Brain Isostasis Monitoring via Triadic Spiral-Time Embeddings

**A Conservative, Falsifiable Framework for Early-Warning Detection in Coupled EEG/ECG Dynamics**

Marcel Krüger¹, Don Feeney²

¹ Independent Researcher, Meiningen, Germany (marcelkrueger092@gmail.com), ORCID: 0009-0002-5709-9729  
² Independent Researcher, Pennsylvania, USA (dfeen87@gmail.com), ORCID: 0009-0003-1350-4160

*January 27, 2026*

---

## Abstract

Cardiac and neural systems form a coupled regulatory unit whose dysregulation can precede acute events such as arrhythmia or seizure-associated autonomic instability. We present a deterministic, operator-based monitoring framework that embeds EEG and ECG/HRV time series into a triadic representation ψ(t) = t + iϕ(t) + jχ(t), where χ(t) := ∂ₜϕ(t) serves as a sensitive marker of nonstationary regime transitions. A preregistered instability functional ΔΦ(t) aggregates spectral/morphological deviation, information-theoretic deviation, and cross-modal coupling deviation under fixed, transparent weights and thresholds.

The framework is explicitly positioned as an interpretable signal-analysis and prospective validation protocol. It does not assert new biophysical laws of physiological time, and it does not prescribe therapeutic intervention or closed-loop control. We define null-test baselines, modality ablations (EEG-only vs. ECG-only vs. coupled), and validation metrics including lead-time distributions and false-alarm rates. The resulting methodology is suitable for preregistered prospective studies and device-level decision support in coupled heart–brain monitoring systems.

---

## 1. Introduction

Cardiac rhythm regulation arises from continuous interaction between myocardial dynamics, autonomic nervous system control, and neurophysiological feedback. Disruptions within this heart–brain regulatory loop can precede acute events such as arrhythmia or seizure-associated autonomic instability, yet often remain undetected by single-modality monitoring approaches.

Recent advances in operator-based analysis of nonstationary time series motivate the development of deterministic, interpretable frameworks that explicitly track regime changes rather than relying on black-box classification. In this work, we introduce a unified instability gate for coupled EEG–ECG monitoring, designed to detect early deviations in spectral structure, information content, and cross-modal coherence within a fully transparent and reproducible pipeline.

### Scope and Safety Statement

This manuscript presents a conservative signal-analysis and prospective validation methodology for early-warning detection. It makes no claims that biological or physiological time is governed by new physical laws, and it does not specify therapeutic stimulation parameters, closed-loop actuation, or implantable device dosing. All statements regarding potential clinical relevance are conditional on prospective, preregistered validation under appropriate ethical and regulatory oversight.

---

## 2. Method Summary

### 2.1 Triadic Embedding

We use the triadic embedding as a mathematical operator on time series:

```
ψ(t) = t + iϕ(t) + jχ(t),    χ(t) := ∂ₜϕ(t)
```

where ϕ(t) is an instantaneous phase (via analytic signal) and χ(t) captures phase torsion / phase acceleration as a regime-change marker.

### 2.2 EEG and ECG/HRV Phase Construction

Let x(t) denote an EEG-derived channel and y(t) denote ECG or HRV time series. Define phases via Hilbert transform H[·]:

```
ϕ_B(t) := arg(H[x(t)]),    χ_B(t) := ∂ₜϕ_B(t)
ϕ_H(t) := arg(H[y(t)]),    χ_H(t) := ∂ₜϕ_H(t)
```

This yields embeddings ψ_B(t) = (t, ϕ_B, χ_B) and ψ_H(t) = (t, ϕ_H, χ_H).

---

## 3. Unified Instability Gate

Relative to a subject-specific baseline window B, define

```
ΔΦ(t) = α|ΔS(t)| + β|ΔI(t)| + γ|ΔC(t)|,    α, β, γ ≥ 0,    α + β + γ = 1
```

where ΔS encodes spectral/morphological deviation, ΔI information/entropy deviation, and ΔC coupling/coherence deviation (EEG–ECG). A deterministic gate is then

```
G(t) = 1{ΔΦ(t) ≥ τ}
```

with threshold τ fixed a priori for prospective evaluation.

---

## 4. ICE Interpretive Layer (non-mechanistic)

We introduce an Information–Coherence–Energy (ICE) interpretation as an organizing view of ΔΦ:

- **Information (I)**: entropy/symbolic complexity (e.g., HRV entropy proxies)
- **Coherence (C)**: phase synchronization (e.g., EEG–ECG coherence)
- **Energy (E)**: load proxies (e.g., LF/HF balance as a conventional proxy)

ICE is not proposed as a new physiological law; it is an interpretive map for established signal features.

![ICE Stability Triangle](docs/images/ice_triangle.png)

*Figure 1: ICE stability triangle as an interpretive visualization of joint deviation space. The diagram provides a conceptual mapping of information, coherence, and energy-related signal features and does not imply a mechanistic physiological model.*

---

## 5. ECG Feature Mapping (Zebra Table)

| Operator/Observable | Conservative Interpretation (Examples) |
|---------------------|----------------------------------------|
| ϕ_H(t) | Beat-to-beat phase stability; rhythm coherence; phase-reset events |
| χ_H(t) = ∂ₜϕ_H(t) | Rapid rhythm-change acceleration; early arrhythmic onset sensitivity |
| ΔS(t) | HRV geometry; QRS/QT morphology drift; spectral redistribution (LF/HF shift) |
| ΔI(t) | Entropy/complexity loss; autonomic rigidity; reduced adaptability regimes |
| ΔC(t) | EEG–ECG decoupling; loss of heart–brain coherence signatures |

*Table 1: Operator features and conservative cardiac interpretation (examples).*

---

## 6. System Architecture (Decision Support, not dosing)

We propose a device-agnostic pipeline for monitoring and early warning, suitable for bedside, wearable, or implant-adjacent telemetry:

1. **Acquisition** (EEG, ECG/HRV), time-sync, artifact rejection
2. **Triadic embedding** and feature extraction
3. **Preregistered gate** G(t), plus ablations (EEG-only/ECG-only/coupled)
4. **Outputs**: risk score, lead-time distribution, and clinician-facing alerts

Any closed-loop neuromodulation is outside scope here and would require separate engineering, clinical trials, and regulatory approval.

```
┌─────────────────────┐         ┌─────────────────────┐
│  EEG Signal         │         │  ECG/HRV Analysis   │
│  Tracking           │         │                     │
│  x(t) → ψ_B(t)      │         │  y(t) → ψ_H(t)      │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           │        ψ_B flow      ψ_H flow│
           │                               │
           └───────────┬───────────────────┘
                       │
           ┌───────────▼────────────┐
           │  Cross-Modal Coupling  │
           │  Coherence ΔC(t)       │
           └───────────┬────────────┘
                       │
           ┌───────────▼────────────┐
           │  HLV Instability Gate  │
           │  ΔΦ(t) ≥ τ             │
           └───────────┬────────────┘
                       │
                       │  Trigger
                       │
           ┌───────────▼────────────┐
           │  Decision Support      │
           │  Risk Scoring & Alerts │
           └────────────────────────┘
```

*Figure 2: Conceptual pipeline for coupled EEG/ECG monitoring and deterministic HLV instability gating. The diagram illustrates data flow and decision-support logic only and does not imply therapeutic intervention or closed-loop stimulation.*

---

## 7. Device-Level Requirements and Safety Constraints

| Aspect | Requirement / Constraint |
|--------|--------------------------|
| **Signal acquisition** | Multi-channel EEG and ECG/HRV with synchronized timestamps; minimum sampling rates consistent with clinical standards |
| **Artifact handling** | Robust rejection of motion, muscle, and electrode artifacts; conservative fallback to "no-decision" state under poor signal quality |
| **Operator computation** | Deterministic, bounded operators only; no adaptive or self-modifying thresholds during deployment |
| **Decision output** | Risk score and alert flags only; no autonomous therapeutic actuation |
| **Fail-safe behavior** | On computational or sensor failure, system defaults to passive monitoring without alerts |
| **False-positive control** | Explicit reporting of false-alarm rate (per hour/day) as a primary performance metric |
| **Clinical interpretability** | All alert conditions traceable to explicit components of ΔΦ(t); no black-box decisions |
| **Failure modes** | Signal dropouts, baseline drift, pharmacological confounds, inter-subject variability; all evaluated explicitly |

*Table 2: Device requirements, safety constraints, and failure modes for operator-based heart–brain monitoring systems.*

---

## 8. Prospective Study Design (Preregistration-ready)

**Design:** Prospective observational or randomized windowing with continuous EEG/ECG. Baseline B defined on stable epochs; evaluation window W fixed. All parameters (α, β, γ, τ) preregistered before unblinding.

**Primary endpoint:** Time-to-instability onset (arrhythmia onset; clinically adjudicated instability; or seizure-associated autonomic instability marker).

**Secondary endpoints:** False alarms per hour; lead time distribution; ablations.

---

## 9. Regulatory and Ethical Considerations

The framework presented here is intended as a decision-support and early-warning methodology. It does not constitute a therapeutic device or clinical intervention.

Any translation into medical hardware or software systems would require compliance with relevant regulatory frameworks, including but not limited to:

- **FDA (United States)**: Software as a Medical Device (SaMD), IEC 62304
- **CE marking (European Union)**: MDR (EU 2017/745)
- **Risk management standards**: ISO 14971
- **Clinical investigation standards**: ISO 14155

All prospective clinical studies must be conducted under institutional review board (IRB) approval, informed consent, and preregistered analysis plans. The present work deliberately avoids specifying stimulation parameters, dosing rules, or implantable device configurations, which fall outside the scope of methodological validation.

---

## 10. Limitations

This framework does not claim mechanistic causation. It is a constrained, falsifiable diagnostic proposal. Failure modes include poor generalization across comorbidities, motion artifacts, and confounding medication effects.

---

## 11. Related Work

Early-warning detection in neurocardiac and neurological systems has been explored in several independent research directions.

### EEG-based seizure forecasting

Multiple studies have demonstrated that phase coherence, entropy measures, and nonstationary spectral features can provide predictive information preceding epileptic seizures [4].

### Heart rate variability and cardiac instability

Reduced HRV complexity and altered spectral balance have been associated with increased risk of arrhythmia and adverse cardiac events [5].

### Closed-loop neuromodulation

Adaptive neurostimulation systems have been proposed for epilepsy and movement disorders, typically relying on black-box classifiers or narrow-band biomarkers [6].

In contrast, the present framework emphasizes a fully interpretable, operator-based instability gate that explicitly separates information, coherence, and energy-related deviations, and admits direct ablation tests across modalities.

---

## 12. Conclusion

We presented an interpretable operator-based instability gate for coupled EEG/ECG monitoring. The approach supports explicit feature mapping, preregistered thresholds, and direct ablation tests of heart–brain coupling terms. Clinical utility must be established prospectively under ethical and regulatory oversight.

---

## Appendix A: Software Architecture, Reproducibility, and Validation

This appendix documents the reference software implementation corresponding to the operator-based heart–brain monitoring framework presented in the main text. The software is designed as a deterministic, modular signal-analysis system intended for research, validation, and decision-support evaluation only.

The implementation follows medical-device-safe language principles: it performs monitoring and early-warning risk estimation, but does not execute therapeutic control, stimulation, or autonomous actuation. All thresholds, parameters, and evaluation criteria are fixed a priori and are explicitly exposed to the user.

**Reference repository:**  
https://github.com/dfeen87/Triadic-Biosignal-Monitor

### A.1 Software Design Philosophy

The software implements the mathematical operators defined in the main manuscript as transparent and testable functions acting on biosignal time series. All components are deterministic, interpretable, and modular, enabling:

- strict ablation studies (EEG-only, ECG-only, coupled),
- reproducible benchmarking and validation,
- separation of signal analysis from any clinical decision or intervention.

No component of the software assumes a biophysical mechanism beyond observable signal features.

### A.2 Repository Structure

The repository is organized as follows:

- **`core/`** — operator definitions and signal processing
  - `phase.py`: Hilbert-based phase extraction, unwrapping, robust derivatives
  - `features.py`: computation of spectral (ΔS), informational (ΔI), and coherence (ΔC) deviation terms
  - `gate.py`: instability functional ΔΦ(t) and thresholding logic
  - `metrics.py`: lead-time, false-alarm rates, ROC curves, ablation metrics
  - `preprocessing.py`: artifact rejection, filtering, synchronization

- **`pipelines/`** — executable analysis pipelines
  - `eeg_only.py`: EEG-only ablation pipeline
  - `ecg_only.py`: ECG-only ablation pipeline
  - `coupled.py`: full coupled EEG–ECG pipeline
  - `streaming.py`: near-real-time streaming mode (Phase 2)

- **`datasets/`** — dataset loaders and synthetic regime-change generators
- **`configs/`** — YAML configuration files defining fixed parameters
- **`notebooks/`** — reproducibility and demonstration notebooks
- **`tests/`** — unit and integration tests
- **`docs/`** — scope, safety, reproducibility, and validation documentation

The figure below summarizes the relationship between the reference repository structure and the analytical pipeline used for validation.

```
┌──────────────────────────────────┐
│     Raw Biosignals               │
│     EEG / ECG / HRV              │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│     Preprocessing                │
│  Artifact rejection, filtering   │
│  preprocessing.py                │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│     Phase Extraction             │
│  Hilbert, unwrap, derivative     │
│  phase.py                        │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│     Feature Computation          │
│     ΔS, ΔI, ΔC                   │
│     features.py                  │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Instability Functional          │
│  ΔΦ(t) and thresholding          │
│  gate.py                         │
└────────────┬─────────────────────┘
             │
     ┌───────┴───────┬─────────────┐
     │               │             │
     ▼               ▼             ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│EEG-only │   │ECG-only │   │ Coupled │
│Pipeline │   │Pipeline │   │Pipeline │
│eeg_only │   │ecg_only │   │coupled  │
│   .py   │   │   .py   │   │   .py   │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     └─────────┬───┴─────────────┘
               │
               ▼
     ┌─────────────────────┐
     │ Evaluation Metrics  │
     │ Lead-time, ROC,     │
     │ false alarms        │
     │ metrics.py          │
     └──────────┬──────────┘
                │
                ▼
     ┌─────────────────────┐
     │ Validation &        │
     │ Reporting           │
     │ run_validation.py   │
     └─────────────────────┘

         ALGORITHM CORE
              ↕
         VALIDATION
```

### A.3 Validation and Ablation Strategy

Validation follows a preregistered and fully deterministic protocol:

- modality ablation analysis (EEG-only vs ECG-only vs coupled),
- synthetic regime-change benchmarks,
- prospective lead-time evaluation,
- reporting of false-alarm rates per unit time,
- sensitivity analysis under fixed parameter configurations.

All reported performance metrics are directly traceable to individual components of the instability functional ΔΦ(t) and admit null-test baselines.

### A.4 Reproducibility and Transparency

Reproducibility is supported through:

- version-controlled source code,
- configuration-driven execution via YAML files,
- executable notebooks reproducing published figures and tables,
- automated test coverage for operators and pipelines.

The software does not rely on proprietary dependencies and can be executed on standard research computing environments.

### A.5 Scope and Safety Statement

The software framework described here constitutes a research and validation tool for early-warning detection and decision support. It does not implement closed-loop neuromodulation, pacing, stimulation control, or autonomous clinical intervention.

Any translation into medical software or hardware systems would require independent engineering validation, prospective clinical trials, and regulatory approval under applicable frameworks such as FDA Software as a Medical Device (SaMD), IEC 62304, ISO 14971, and EU MDR 2017/745.

---

## Data and Code Availability

Reference implementations and evaluation scripts are available in the `triadic-biosignal-monitor` repository:

**Repository:** https://github.com/dfeen87/Triadic-Biosignal-Monitor

Clinical deployment requires local approvals and appropriate data governance.

---

## References

[1] M. Krüger, *Information-Theoretic Modeling of Neural Coherence via Triadic Spiral-Time Dynamics: A Framework for Neurodynamic Collapse*, Zenodo (2026). doi: [10.5281/zenodo.18213517](https://doi.org/10.5281/zenodo.18213517)

[2] N. W. (repository), *NeuroDynamics Collapse Validation: EEG Part Four* (notebook), https://github.com/nwycomp/NeuroDynamics-Collapse-Validation-/blob/main/eeg-part-four.ipynb

---

## Appendix: Conceptual Pipeline Diagram

```
┌──────────┐                    ┌─────────────┐
│   EEG    │                    │  ECG/HRV    │
│  x(t)    │                    │   y(t)      │
└────┬─────┘                    └──────┬──────┘
     │                                 │
     │ Phase extraction                │ Phase extraction
     │ via Hilbert                     │ via Hilbert
     ▼                                 ▼
┌────────────┐                  ┌─────────────┐
│  ψ_B(t)    │                  │  ψ_H(t)     │
│ (t,ϕ_B,χ_B)│                  │ (t,ϕ_H,χ_H) │
└─────┬──────┘                  └──────┬──────┘
      │                                │
      └────────────┬───────────────────┘
                   │
                   │ Compute coupling
                   │ coherence ΔC(t)
                   ▼
          ┌────────────────┐
          │ Instability    │
          │ Gate           │
          │ ΔΦ(t) ≥ τ      │
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │ Decision       │
          │ Support        │
          │ risk score /   │
          │ alerts         │
          └────────────────┘
```

*Figure 3: Conceptual pipeline for coupled EEG/ECG monitoring and deterministic instability gating.*

---

**Document Version:** 1.0  
**Last Updated:** January 27, 2026  
**Status:** Preprint / Under Review

---

## Citation

```bibtex
@article{kruger2026operator,
  author = {Krüger, Marcel and Feeney, Don},
  title = {Operator-Based Heart–Brain Isostasis Monitoring via 
           Triadic Spiral-Time Embeddings: A Conservative, Falsifiable 
           Framework for Early-Warning Detection in Coupled EEG/ECG Dynamics},
  year = {2026},
  note = {Preprint}
}
```
