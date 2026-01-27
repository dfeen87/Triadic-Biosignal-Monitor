# Validation Protocol

**Operator-Based Heart-Brain Monitoring Framework**  
**Prospective Validation Guidelines**  
**Version 1.0**  
**Last Updated: January 27, 2026**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Study Design Framework](#study-design-framework)
3. [Preregistration Requirements](#preregistration-requirements)
4. [Participant Selection](#participant-selection)
5. [Data Collection Protocol](#data-collection-protocol)
6. [Signal Processing Pipeline](#signal-processing-pipeline)
7. [Statistical Analysis Plan](#statistical-analysis-plan)
8. [Performance Metrics](#performance-metrics)
9. [Safety Monitoring](#safety-monitoring)
10. [Reporting Standards](#reporting-standards)
11. [Sample Protocol Template](#sample-protocol-template)
12. [Regulatory Considerations](#regulatory-considerations)

---

## Introduction

### Purpose

This document provides guidelines for conducting prospective validation studies of the operator-based heart-brain monitoring framework. These guidelines are designed to ensure:

- **Scientific rigor** through preregistered protocols
- **Reproducibility** via standardized procedures
- **Safety** through appropriate oversight
- **Regulatory readiness** for potential device translation

### Scope

These guidelines apply to:

✓ Prospective observational studies  
✓ Controlled validation trials  
✓ Multi-center validation studies  
✓ Algorithm comparison studies  

These guidelines do NOT cover:

✗ Retrospective analyses (different standards apply)  
✗ Therapeutic intervention trials (require separate IND/IDE)  
✗ Unsupervised home monitoring (not yet validated)  

### Target Audience

- Clinical researchers planning validation studies
- Institutional Review Boards (IRBs) reviewing protocols
- Regulatory scientists evaluating methodologies
- Data Safety Monitoring Boards (DSMBs)

---

## Study Design Framework

### Design Options

#### Option 1: Prospective Observational Study

**Description:** Monitor participants with standard-of-care + research system in parallel

**Advantages:**
- Lower risk to participants
- Easier IRB approval
- Can collect large datasets
- Real-world performance data

**Disadvantages:**
- Cannot assess clinical utility directly
- Requires adjudicated ground truth
- May miss rare events

**Recommended for:** Initial validation, algorithm refinement

**Example hypothesis:**
> "The operator-based framework detects preictal changes ≥5 minutes before seizure onset in ≥70% of events with false alarm rate <1 per hour."

---

#### Option 2: Randomized Windowing Study

**Description:** Randomize analysis window parameters while maintaining identical data collection

**Advantages:**
- Tests parameter sensitivity
- Controls for confounders
- Maintains safety (no intervention)
- Allows within-subject comparisons

**Disadvantages:**
- Requires careful randomization
- May need larger sample size
- Complex statistical analysis

**Recommended for:** Threshold optimization, ablation validation

**Example design:**
> Randomize baseline duration (20s vs. 30s vs. 40s) across participants, compare detection performance.

---

#### Option 3: Controlled Comparison Study

**Description:** Compare operator-based framework against established methods

**Advantages:**
- Demonstrates relative performance
- Benchmark against state-of-art
- Facilitates publication
- Informs clinical value

**Disadvantages:**
- Requires validated comparator
- May need multiple comparisons correction
- Longer data collection

**Recommended for:** Superiority/non-inferiority claims

**Example comparison:**
> Operator-based framework vs. simple amplitude threshold vs. spectral power trend analysis

---

### Sample Size Considerations

**Primary endpoint: Detection sensitivity**

Use this formula for sensitivity estimation:

```
n = (Z_α + Z_β)² × [p(1-p)] / d²

Where:
- Z_α = 1.96 (for α=0.05, two-tailed)
- Z_β = 0.84 (for 80% power)
- p = expected sensitivity (e.g., 0.75)
- d = desired precision (e.g., 0.10)

Example:
n = (1.96 + 0.84)² × [0.75 × 0.25] / 0.10²
n = 7.84 × 0.1875 / 0.01
n ≈ 147 events needed
```

**Secondary endpoint: False alarm rate**

For false alarm rate per hour:

```
Total monitoring hours = (desired_CI_width / (2 × Z_α/2)) × √(expected_FAR / hours)

Example for ±0.5 alarms/hour CI:
Hours ≈ 500-1000 hours of interictal monitoring
```

**Practical considerations:**

- **Pilot studies:** Start with n=10-20 participants
- **Multi-center:** Account for site variability (increase by 20-30%)
- **Dropout:** Add 15-20% for expected attrition
- **Rare events:** May need 50-100 participants for adequate event count

---

## Preregistration Requirements

### Why Preregister?

Preregistration prevents:
- Data dredging and p-hacking
- Selective outcome reporting
- Post-hoc hypothesis generation
- Publication bias

**Requirement:** All prospective studies MUST preregister before data unblinding.

### Where to Preregister

**Recommended registries:**

1. **ClinicalTrials.gov** (for clinical studies)
   - Required for FDA submission
   - Public registry
   - NCT number assigned

2. **Open Science Framework (OSF)**
   - Free and open
   - Time-stamped preregistration
   - Supports detailed statistical plans
   - https://osf.io/

3. **AsPredicted.org**
   - Simple interface
   - Quick preregistration
   - https://aspredicted.org/

### Required Preregistration Elements

#### 1. Study Identification

```
Study title: [Full descriptive title]
Principal Investigator: [Name, affiliation]
Co-investigators: [Names, affiliations]
Registration date: [Date before data collection]
Expected completion: [Projected end date]
```

#### 2. Hypotheses (Prespecified)

**Primary hypothesis:**
> The operator-based framework detects [target event] with sensitivity ≥[X]% and false alarm rate ≤[Y] per hour.

**Secondary hypotheses:**
- Coupled mode superiority over single-modality
- Lead time distribution characteristics
- Subgroup analyses (if any)

**Example:**
```
H1 (Primary): Framework achieves ≥70% sensitivity for preictal detection
H2 (Secondary): Mean lead time ≥5 minutes before seizure onset
H3 (Exploratory): Coupled mode outperforms EEG-only by ≥15%
```

#### 3. Study Design

```
Design type: [Observational / Randomized / Comparative]
Blinding: [Single / Double / Open-label]
Randomization: [Method if applicable]
Control condition: [Standard monitoring / Alternative algorithm]
Study duration: [Per participant and total]
```

#### 4. Participant Criteria

**Inclusion criteria:**
- Age range
- Diagnosis
- Seizure/event frequency
- Medication status
- Able to provide informed consent

**Exclusion criteria:**
- Comorbidities that would confound results
- Medications that alter EEG/ECG
- Skin conditions preventing electrode placement
- Pregnancy (if applicable)
- Unable to tolerate monitoring duration

#### 5. Sample Size Justification

```
Target N: [Number]
Calculation: [Formula and assumptions]
Power: [Usually 80% or 90%]
Alpha: [Usually 0.05]
Expected effect size: [Based on pilot or literature]
Dropout assumption: [Percentage]
```

#### 6. Data Collection Procedures

**Specify exactly:**
- EEG montage (which channels)
- ECG lead configuration
- Sampling rates
- Recording duration per session
- Environment (lab/clinic/EMU)
- Parallel standard monitoring
- Event adjudication process

#### 7. Analysis Plan (LOCKED)

**Primary analysis:**
- Statistical test (e.g., binomial test for sensitivity)
- Significance level (α = 0.05)
- Confidence intervals (95%)
- Handling of missing data

**Secondary analyses:**
- Lead-time analysis (survival curves, Kaplan-Meier)
- False alarm rate (Poisson regression)
- Subgroup analyses (if prespecified)

**Exploratory analyses:**
- Clearly labeled as exploratory
- Not used for primary claims

#### 8. Preprocessing Parameters (FIXED)

```yaml
# All parameters must be preregistered:
baseline_duration: 30  # seconds
window_size: 15        # seconds
step_size: 3           # seconds
eeg_bandpass: [0.5, 50]  # Hz
ecg_bandpass: [0.5, 40]  # Hz
artifact_threshold: 5.0   # standard deviations

# Instability gate parameters:
alpha: 0.4    # Spectral weight
beta: 0.3     # Information weight
gamma: 0.3    # Coupling weight
threshold: 2.0  # Decision boundary (std dev)
```

**CRITICAL:** No parameter tuning after unblinding!

#### 9. Stopping Rules

**Prespecify conditions for early termination:**

**Safety stopping rules:**
- Serious adverse event related to monitoring
- Equipment malfunction causing participant risk
- Unanticipated harm to participants

**Futility stopping rules:**
- Interim analysis shows <5% chance of meeting primary endpoint
- Enrollment challenges make completion infeasible

**Success stopping rules (optional):**
- Primary endpoint met with overwhelming evidence (p < 0.001) at interim

#### 10. Data Sharing Plan

```
Data availability: [Public / Restricted / Upon request]
Repository: [e.g., Zenodo, Dryad, PhysioNet]
Embargo period: [If any]
De-identification method: [HIPAA Safe Harbor / Expert determination]
Code availability: [GitHub repository]
```

---

## Participant Selection

### Eligibility Criteria

#### For Seizure Detection Studies

**Inclusion criteria:**
- Age 18-65 years
- Diagnosis of epilepsy per ILAE criteria
- ≥1 seizure per month (average over past 3 months)
- On stable antiepileptic medication (≥4 weeks)
- Able to provide informed consent
- Willing to undergo continuous monitoring

**Exclusion criteria:**
- Psychogenic non-epileptic seizures (PNES) only
- Status epilepticus within past 6 months
- Cardiac arrhythmia requiring treatment
- Pacemaker or implanted defibrillator
- Skin conditions preventing electrode placement
- Pregnancy or nursing
- Non-English speaking (if consent materials in English only)

#### For Cardiac Arrhythmia Studies

**Inclusion criteria:**
- Age 18-75 years
- History of documented arrhythmia
- Able to undergo extended Holter monitoring
- Stable cardiac medications (≥4 weeks)
- Normal baseline EEG (no seizure disorder)

**Exclusion criteria:**
- Recent myocardial infarction (<3 months)
- Severe heart failure (NYHA Class IV)
- Uncontrolled hypertension (SBP >180 or DBP >110)
- Recent stroke (<6 months)
- Concurrent epilepsy diagnosis

### Recruitment Strategy

**Sources:**
- Epilepsy Monitoring Unit (EMU) admissions
- Cardiology clinic referrals
- Neurology clinic patients
- Research registries
- Community advertisements (with IRB approval)

**Screening process:**
1. Initial phone screen (5-10 minutes)
2. Review of medical records
3. In-person eligibility assessment
4. Informed consent (allow ≥24 hours for decision)
5. Enrollment

**Consent elements:**
- Purpose: Research validation of monitoring algorithm
- Duration: [Specify hours/days]
- Procedures: EEG and ECG electrode placement, continuous monitoring
- Risks: Skin irritation, discomfort, breach of privacy
- Benefits: None direct; contribution to research
- Compensation: [Specify amount/method]
- Right to withdraw at any time
- Data confidentiality protections

---

## Data Collection Protocol

### Equipment Requirements

#### EEG Acquisition

**Minimum specifications:**
- **Channels:** ≥10 (international 10-20 system)
- **Sampling rate:** ≥250 Hz
- **Resolution:** ≥16 bits
- **Common montage:** Bipolar or average reference
- **Impedance:** <5 kΩ at start of recording

**Recommended channels:**
- F7, F8, T3, T4, T5, T6, O1, O2 (temporal lobe)
- Fz, Cz, Pz (midline)
- Plus ground and reference

#### ECG Acquisition

**Minimum specifications:**
- **Leads:** Single lead (II) minimum; 3-lead preferred
- **Sampling rate:** ≥250 Hz (synchronized with EEG)
- **Resolution:** ≥16 bits
- **Filters:** Baseline wander correction, 60 Hz notch

**Lead placement:**
- Lead II: Right arm (-), Left leg (+)
- Or standard 3-lead chest configuration

#### Synchronization

**CRITICAL:** EEG and ECG must be synchronized to within 10 ms.

**Methods:**
- Hardware synchronization (same acquisition system)
- Software timestamping (NTP synchronized)
- Sync pulse injection (for separate systems)

### Recording Procedure

#### Session Timeline

```
Time 0:    Participant arrival, final consent verification
+15 min:   Skin preparation, electrode placement
+30 min:   Impedance check, system test
+45 min:   Baseline recording begins (30 minutes minimum)
+75 min:   Monitoring session start
...        [Continuous monitoring for study duration]
-15 min:   Final check, prepare for electrode removal
End:       Electrode removal, skin care, compensation
```

#### Quality Control During Recording

**Every 30 minutes:**
- [ ] Check electrode impedances (<5 kΩ)
- [ ] Verify data streaming correctly
- [ ] Review signal quality visually
- [ ] Document any artifacts or events
- [ ] Check participant comfort

**If signal quality degrades:**
1. Pause research recording (standard monitoring continues)
2. Identify problem (loose electrode, motion artifact, etc.)
3. Rectify issue (re-gel, re-position)
4. Resume recording
5. Document interruption in log

### Event Documentation

#### For Seizure Studies

**Standardized event log:**

```
Event ID: [Sequential number]
Date/Time: [Timestamp]
Event type: [Focal / Generalized / Other]
Duration: [Seconds]
Severity: [1-5 scale or validated scale]
Clinical manifestations: [Free text]
EEG onset: [Timestamp]
Clinical onset: [Timestamp]
Witnessed by: [Clinician initials]
Video available: [Yes/No]
```

**Adjudication process:**
- Two independent board-certified epileptologists review
- Blind to algorithm output
- Resolve disagreements via consensus or third reviewer
- Document final determination

#### For Cardiac Studies

**Event log:**

```
Event ID: [Sequential number]
Date/Time: [Timestamp]
Arrhythmia type: [AF / VT / SVT / Other]
Duration: [Seconds]
Hemodynamically stable: [Yes/No]
Symptoms reported: [Yes/No]
ECG onset: [Timestamp]
Symptomatic onset: [Timestamp if applicable]
Cardiologist confirmation: [Initials]
```

---

## Signal Processing Pipeline

### Preprocessing Standards

#### Artifact Rejection

**Automated rejection criteria:**
- Amplitude >5 standard deviations from baseline
- Sudden impedance change (electrode pop)
- Flat signal (equipment failure)
- High-frequency noise (>100 Hz power spike)

**Manual review:**
- Technician reviews all auto-flagged segments
- Document reason for rejection
- Maximum 20% data loss acceptable per session

#### Filtering

**Standardized filters (preregistered):**

```python
# EEG bandpass: 0.5-50 Hz
eeg_filtered = butter_bandpass(eeg_raw, 0.5, 50, fs=250, order=4)

# ECG bandpass: 0.5-40 Hz
ecg_filtered = butter_bandpass(ecg_raw, 0.5, 40, fs=250, order=4)

# Notch filter: 60 Hz (or 50 Hz) if necessary
signal = notch_filter(signal, freq=60, Q=30, fs=250)
```

**NO additional filtering post-hoc without preregistration amendment.**

### Feature Extraction

**Windowing parameters (FIXED):**

```python
config = {
    'baseline_duration': 30,  # seconds, first 30s of recording
    'window_size': 15,        # seconds
    'step_size': 3,           # seconds (80% overlap)
    'fs': 250                 # Hz
}
```

**Features computed per window:**

| Feature | Modality | Computation |
|---------|----------|-------------|
| ΔS | EEG | Z-score of alpha/beta ratio vs. baseline |
| ΔI | EEG | Z-score of permutation entropy vs. baseline |
| ΔS | ECG | Z-score of LF/HF ratio vs. baseline |
| ΔI | ECG | Z-score of variance vs. baseline |
| ΔC | Both | Z-score of phase sync + coherence vs. baseline |

### Instability Gate

**Unified functional (PREREGISTERED):**

```python
delta_phi = alpha * delta_S + beta * delta_I + gamma * delta_C

# Weights (FIXED):
alpha = 0.4  # Spectral
beta = 0.3   # Information
gamma = 0.3  # Coupling

# Threshold (FIXED):
tau = 2.0  # standard deviations

# Gate:
G = 1 if delta_phi >= tau else 0
```

**Alert generation:**
- Consecutive windows required: 3 (persistence)
- Cooldown period: 60 seconds (avoid repeated alerts)

### Blinding Procedure

**CRITICAL for prospective validation:**

1. **Real-time processing:** Algorithm runs during recording
2. **Output hidden:** Results saved but NOT shown to clinicians
3. **Unblinding:** Only after:
   - All data collected
   - All events adjudicated by blinded clinicians
   - Analysis plan finalized
4. **Unblinding performed by:** Independent statistician

**Maintains integrity of:**
- Event adjudication (unbiased)
- Clinical care (not influenced by research system)
- Statistical analysis (prevents data-dependent decisions)

---

## Statistical Analysis Plan

### Primary Outcome Analysis

#### Detection Sensitivity

**Null hypothesis:**
> H₀: Sensitivity ≤ 50% (chance level)

**Alternative hypothesis:**
> H₁: Sensitivity > 70% (clinically meaningful)

**Statistical test:**
- One-sample binomial test
- α = 0.05 (one-tailed)
- Calculate exact p-value

**Example R code:**

```r
# Observed: 45 detections out of 60 events
binom.test(x = 45, n = 60, p = 0.50, alternative = "greater")

# Result interpretation:
# p < 0.05 → Reject H₀, conclude sensitivity significantly > 50%
```

**Confidence interval:**

```r
# 95% CI for sensitivity
binom.test(x = 45, n = 60, p = 0.50, conf.level = 0.95)$conf.int
# [0.623, 0.865]
```

#### Lead Time Analysis

**Metric:** Time from first alert to event onset (minutes)

**Analysis:**
1. **Survival analysis:** Kaplan-Meier curves for lead time
2. **Median and IQR:** Report median lead time with quartiles
3. **Comparison:** If multiple algorithms, log-rank test

**Example:**

```r
library(survival)

# Data: lead_time (minutes), detected (1/0)
surv_obj <- Surv(time = lead_time, event = detected)
fit <- survfit(surv_obj ~ 1)

# Plot
plot(fit, xlab = "Lead time (minutes)", 
     ylab = "Proportion detected",
     main = "Kaplan-Meier: Detection Lead Time")

# Median lead time
quantile(fit, probs = 0.5)
```

### Secondary Outcome Analysis

#### False Alarm Rate

**Metric:** False alarms per hour of interictal monitoring

**Analysis:**
- Poisson regression (if covariates)
- Or simple rate calculation with 95% CI

**Example:**

```r
# Observed: 15 false alarms in 500 hours
fa_rate <- 15 / 500  # = 0.03 per hour

# 95% CI using Poisson
poisson.test(x = 15, T = 500)$conf.int
# [0.017, 0.049] per hour
```

#### Ablation Comparison

**Compare:** EEG-only vs. ECG-only vs. Coupled

**Statistical test:**
- Repeated measures ANOVA (if all modes tested on same subjects)
- Or paired t-tests with Bonferroni correction

**Example:**

```r
# Lead times for three modes (same subjects)
eeg_only <- c(5, 8, 6, 10, 7, ...)
ecg_only <- c(3, 5, 4, 6, 5, ...)
coupled <- c(12, 15, 14, 18, 16, ...)

# Repeated measures ANOVA
data <- data.frame(
  subject = rep(1:n, 3),
  mode = rep(c("EEG", "ECG", "Coupled"), each = n),
  lead_time = c(eeg_only, ecg_only, coupled)
)

aov_result <- aov(lead_time ~ mode + Error(subject/mode), data = data)
summary(aov_result)

# Post-hoc pairwise comparisons
pairwise.t.test(data$lead_time, data$mode, p.adjust.method = "bonferroni")
```

### Subgroup Analysis (If Prespecified)

**Potential subgroups:**
- Seizure type (focal vs. generalized)
- Medication class
- Disease duration
- Age group

**Statistical approach:**
- Interaction tests (subgroup × detection)
- Stratified analysis
- **Correct for multiple comparisons** (Bonferroni or FDR)

**Example:**

```r
# Test if sensitivity differs by seizure type
fisher.test(table(detected, seizure_type))

# If significant, report separately:
sensitivity_focal <- sum(detected[focal]) / length(focal)
sensitivity_generalized <- sum(detected[generalized]) / length(generalized)
```

### Handling Missing Data

**Prespecified rules:**

1. **Complete case analysis** (primary)
   - Exclude participants with >20% data loss
   - Report attrition in CONSORT diagram

2. **Sensitivity analysis** (secondary)
   - Multiple imputation (if missingness <10%)
   - Last observation carried forward (LOCF) for dropouts
   - Compare results to complete case

**Missing event adjudication:**
- If event cannot be adjudicated → exclude from denominator
- Document reason for each exclusion
- Report in results (e.g., "5 events excluded due to poor video quality")

---

## Performance Metrics

### Primary Metrics

#### 1. Sensitivity (True Positive Rate)

**Definition:**
```
Sensitivity = TP / (TP + FN)

Where:
- TP = True positives (correctly detected events)
- FN = False negatives (missed events)
```

**Target:** ≥70% for clinical utility

**95% CI:** Use Wilson score interval

#### 2. False Alarm Rate

**Definition:**
```
FAR = Number of false alarms / Total interictal hours

Units: per hour
```

**Target:** <1 per hour for clinical acceptability

**95% CI:** Poisson exact

#### 3. Lead Time

**Definition:**
```
Lead time = Time of first alert - Time of event onset

Units: seconds or minutes
Positive = early warning
Negative = late detection
```

**Target:** Median ≥5 minutes (300 seconds)

**Analysis:** Kaplan-Meier survival curves

### Secondary Metrics

#### 4. Specificity (True Negative Rate)

**Definition:**
```
Specificity = TN / (TN + FP)
```

**Note:** Often not directly applicable to continuous monitoring (no defined "negative" windows). Use false alarm rate instead.

#### 5. Positive Predictive Value (PPV)

**Definition:**
```
PPV = TP / (TP + FP)
```

**Depends on event prevalence**—report with prevalence.

#### 6. Area Under ROC Curve (AUC)

**Definition:** Discriminability across all thresholds

**Computation:**
```r
library(pROC)
roc_obj <- roc(actual_labels, predicted_scores)
auc(roc_obj)
```

**Interpretation:**
- AUC = 0.5: Chance
- AUC > 0.7: Acceptable
- AUC > 0.8: Excellent
- AUC > 0.9: Outstanding

### Exploratory Metrics

#### 7. Time to First False Alarm

**Definition:** Duration of monitoring before first false alarm

**Analysis:** Survival analysis (Kaplan-Meier)

#### 8. Alert Clustering

**Definition:** Temporal distribution of alerts

**Analysis:** Autocorrelation, inter-alert interval histogram

#### 9. Signal Quality Impact

**Definition:** Relationship between signal quality and performance

**Analysis:** Regression of detection accuracy on artifact ratio

---

## Safety Monitoring

### Data Safety Monitoring Board (DSMB)

**When required:**
- Multi-site studies
- Studies >50 participants
- Studies with intervention component
- Required by sponsor/funder

**Composition:**
- Independent clinician (not study team)
- Statistician
- Ethicist or patient advocate

**Responsibilities:**
- Review adverse events
- Monitor data quality
- Recommend continuation/modification/termination
- Meets every 6-12 months (or per charter)

### Adverse Event Monitoring

**Definition of adverse event (AE):**
> Any unfavorable medical occurrence in a participant, whether or not related to the research procedures.

**Classification:**

| Severity | Definition | Examples |
|----------|------------|----------|
| Mild | Minimal discomfort, no intervention needed | Skin redness from electrodes |
| Moderate | Moderate discomfort, minor intervention | Electrode site dermatitis requiring cream |
| Severe | Significant discomfort, medical intervention | Allergic reaction to electrode paste |
| Life-threatening | Immediate risk of death | Not expected from monitoring |
| Death | Participant death | Report immediately |

**Causality assessment:**

- **Definitely related:** AE caused by research procedures
- **Probably related:** Likely caused by research procedures
- **Possibly related:** May be caused by research procedures
- **Unlikely related:** Probably not caused by research procedures
- **Not related:** Clearly from other cause

**Reporting timelines:**

- **Serious AE (SAE):** Report to IRB within 24-48 hours
- **Unexpected SAE:** Report to IRB and sponsor immediately
- **Non-serious AE:** Report in annual continuing review
- **All AEs:** Logged in study database

### Participant Withdrawal

**Participants may withdraw for:**
- Personal reasons (no explanation required)
- Adverse event
- Protocol violation
- Lost to follow-up
- Investigator decision (safety concern)

**Upon withdrawal:**
1. Document reason in case report form (CRF)
2. Collect final data if participant consents
3. Offer continued clinical care
4. Include in intention-to-treat analysis (if any data collected)
5. Do not replace participant unless prespecified

---

## Reporting Standards

### CONSORT-Style Flow Diagram

```
Assessed for eligibility (n = XXX)
    |
    ├─ Excluded (n = XX)
    │   ├─ Did not meet inclusion criteria (n = X)
    │   ├─ Declined to participate (n = X)
    │   └─ Other reasons (n = X)
    |
Enrolled (n = XX)
    |
    ├─ Received monitoring (n = XX)
    │
    ├─ Discontinued (n = X)
    │   ├─ Adverse event (n = X)
    │   ├─ Withdrew consent (n = X)
    │   ├─ Lost to follow-up (n = X)
    │   └─ Other (n = X)
    |
Analyzed (n = XX)
    |
    ├─ Excluded from analysis (n = X)
    │   └─ Reason: [specify]
```

### Results Reporting Checklist

**Participant characteristics:**
- [ ] Demographics table (age, sex, race/ethnicity)
- [ ] Clinical characteristics (diagnosis, duration, medications)
- [ ] Baseline measurements (if applicable)

**Primary outcome:**
- [ ] Sensitivity with 95% CI
- [ ] False alarm rate with 95% CI
- [ ] Lead time: median, IQR, range
- [ ] Primary hypothesis test result (p-value)

**Secondary outcomes:**
- [ ] Ablation comparison results
- [ ] Subgroup analyses (if prespecified)
- [ ] Exploratory findings (labeled as such)

**Safety:**
- [ ] Adverse events table
- [ ] Serious adverse events (narrative)
- [ ] Withdrawals due to AE

**Quality:**
- [ ] Data completeness (% missing)
- [ ] Protocol deviations
- [ ] Signal quality metrics

### Figure and Table Standards

**Required figures:**
1. CONSORT flow diagram
2. Primary outcome visualization (e.g., ROC curve)
3. Lead time distribution (histogram or Kaplan-Meier)
4. Representative signal example with detection

**Required tables:**
1. Participant baseline characteristics
2. Primary outcome results
3. Adverse events summary

### Publication Guidelines

**Authorship:**
- Follow ICMJE criteria (substantial contributions, drafting/revising, final approval, accountability)
- Acknowledge contributors who don't meet authorship criteria

**Data sharing:**
- Make de-identified data available per preregistration plan
- Deposit in public repository (e.g., PhysioNet, Zenodo)
- Share analysis code (GitHub)

**Preprint:**
- Consider posting preprint (e.g., medRxiv, bioRxiv)
- Note: Some journals have preprint policies

**Peer review:**
- Submit to peer-reviewed journal
- Respond to reviewer comments transparently
- Make revisions clear in tracked changes

---

## Sample Protocol Template

### Title Page

```
PROTOCOL TITLE: Prospective Validation of Operator-Based Heart-Brain 
                Monitoring for Early Detection of Seizures

Protocol Version: 1.0
Protocol Date: [Date]

Principal Investigator:
  Name: [PI Name]
  Institution: [Institution]
  Email: [Email]
  Phone: [Phone]

Co-Investigators:
  [List all]

Sponsor: [If applicable]

IRB: [Institution] IRB
Protocol Number: [Assigned after submission]
ClinicalTrials.gov ID: [NCT number]
```

### Protocol Synopsis

```
Study Phase: Observational validation study

Study Design: Prospective, single-center, observational

Primary Objective: 
  Validate the operator-based framework for early detection of 
  seizures in adults with epilepsy

Secondary Objectives:
  1. Characterize lead-time distribution
  2. Determine false alarm rate
  3. Compare coupled vs. single-modality performance

Study Population: 
  Adults (18-65 years) with epilepsy, ≥1 seizure/month

Sample Size: 
  N = 40 participants, target 80 observed seizures

Study Duration: 
  Enrollment: 12 months
  Per participant: 24-72 hours of monitoring
  Total study: 18 months

Primary Endpoint: 
  Detection sensitivity (%)

Key Inclusion Criteria:
  - Age 18-65
  - Epilepsy diagnosis per ILAE
  - ≥1 seizure/month
  - Able to consent

Key Exclusion Criteria:
  - Cardiac arrhythmia
  - Pacemaker/ICD
  - Pregnancy
  - Status epilepticus <6 months

Statistical Analysis:
  Primary: One-sample binomial test for sensitivity
  Secondary: Kaplan-Meier for lead time, Poisson for FAR
  Significance: α = 0.05

Ethical Approval: 
  [Institution] IRB approval obtained [Date]

Funding: 
  [Source if applicable]
```

### Full Protocol Sections

**1. Background and Rationale**
- Brief literature review
- Gaps in current monitoring
- Rationale for this study
- Preliminary data (if available)

**2. Study Objectives**
- Primary objective (one clear statement)
- Secondary objectives (2-4 additional aims)
- Exploratory objectives (hypothesis-generating)

**3. Study Design**
- Design type
- Study schematic (flow diagram)
- Duration and timeline
- Setting (EMU, clinic, etc.)

**4. Study Population**
- Inclusion criteria (detailed)
- Exclusion criteria (detailed)
- Justification for criteria
- Vulnerable populations (if applicable)

**5. Recruitment and Consent**
- Recruitment methods
- Screening procedures
- Informed consent process
- Compensation details

**6. Study Procedures**
- Visit schedule
- Baseline assessments
- Monitoring procedures
- Data collection forms
- Follow-up (if applicable)

**7. Intervention (if applicable)**
- N/A for observational studies
- Include if any intervention component

**8. Study Endpoints**
- Primary endpoint definition
- Secondary endpoints
- Exploratory endpoints
- Event adjudication process

**9. Statistical Considerations**
- Sample size calculation
- Analysis plan (primary, secondary, exploratory)
- Handling of missing data
- Interim analyses (if planned)
- Multiple comparison corrections

**10. Data Management**
- Data collection systems
- Data entry and validation
- Data storage and security
- Quality control procedures
- Database lock procedures

**11. Safety Monitoring**
- AE definitions
- Reporting procedures
- DSMB (if applicable)
- Stopping rules

**12. Ethical Considerations**
- IRB approval status
- Informed consent
- Privacy and confidentiality
- Risk-benefit assessment
- Vulnerable populations

**13. Publication Plan**
- Authorship criteria
- Data sharing
- Results dissemination
- Timeline for publication

**14. References**
- Key citations
- Regulatory guidance documents

**15. Appendices**
- Informed consent form
- Case report forms
- Data collection instruments
- Analysis code/scripts

---

## Regulatory Considerations

### IRB Submission Checklist

**Required documents:**
- [ ] Protocol (this document)
- [ ] Informed consent form
- [ ] HIPAA authorization (if US-based)
- [ ] Recruitment materials
- [ ] Case report forms
- [ ] Investigator brochure or device manual
- [ ] PI CV/training documentation
- [ ] Conflict of interest disclosures
- [ ] Letters of support (if multi-site)

**IRB review type:**
- Expedited (if minimal risk, observational)
- Full board (if >minimal risk, vulnerable populations)

**Expected timeline:**
- Expedited: 2-4 weeks
- Full board: 4-8 weeks

### Regulatory Authority Interaction

**FDA (if applicable):**
- Investigational Device Exemption (IDE) NOT required for non-significant risk device studies
- Consult FDA early if unsure about risk classification

**EU (if applicable):**
- Clinical investigation per MDR Annex XV
- Ethics committee approval required
- Competent authority notification (if applicable)

### Data Protection

**HIPAA (United States):**
- De-identify per Safe Harbor or Expert Determination
- Limited data set with Data Use Agreement (if needed)
- HIPAA authorization from participants

**GDPR (European Union):**
- Lawful basis for processing (consent or research exemption)
- Data protection impact assessment (DPIA)
- Data processing agreement with processors
- Right to withdraw consent and data deletion

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-27 | Initial validation protocol guidelines |

---

## Contact for Protocol Questions

**Principal Authors:**
- Marcel Krüger: marcelkrueger092@gmail.com
- Don Feeney: dfeen87@gmail.com

**For study-specific questions:**
- Contact your site Principal Investigator
- CC study coordinating center (if multi-site)

---

## Acknowledgments

This validation protocol template was developed with input from:
- Clinical research coordinators
- Biostatisticians
- IRB chairs
- Regulatory affairs specialists
- Patient advocates

We thank all contributors for their expertise.

---

**END OF VALIDATION PROTOCOL**
