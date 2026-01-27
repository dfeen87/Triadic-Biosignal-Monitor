# Scope and Safety Statement

**Operator-Based Heart-Brain Monitoring Framework**  
**Version 1.0**  
**Last Updated: January 27, 2026**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scope of This Work](#scope-of-this-work)
3. [What This Framework IS](#what-this-framework-is)
4. [What This Framework IS NOT](#what-this-framework-is-not)
5. [Safety Constraints](#safety-constraints)
6. [Regulatory Considerations](#regulatory-considerations)
7. [Limitations and Known Issues](#limitations-and-known-issues)
8. [Ethical Considerations](#ethical-considerations)
9. [Responsible Use Guidelines](#responsible-use-guidelines)
10. [Contact and Reporting](#contact-and-reporting)

---

## Executive Summary

This repository contains a **research validation framework** for early-warning detection in coupled EEG-ECG monitoring systems. It implements deterministic, operator-based signal analysis methods designed for prospective validation studies.

**Critical Safety Statement:**

> This software is a **research tool** for signal analysis and decision support validation. It does **NOT** constitute a medical device, does **NOT** provide therapeutic intervention, and is **NOT** approved for clinical diagnosis or treatment. All use must be under appropriate research ethics approval with informed consent.

---

## Scope of This Work

### 1. Primary Purpose

This framework provides:

- **Signal analysis methodology** for detecting regime transitions in biosignals
- **Validation protocols** for prospective early-warning studies
- **Reference implementation** of operator-based instability detection
- **Reproducible benchmarks** for biosignal monitoring research

### 2. Intended Users

- **Biomedical researchers** conducting prospective validation studies
- **Signal processing scientists** developing early-warning systems
- **Clinical research teams** with appropriate ethics approval
- **Regulatory scientists** evaluating monitoring methodologies

### 3. Intended Use Cases

✓ **Research validation studies** with preregistered protocols  
✓ **Algorithm benchmarking** against synthetic and annotated datasets  
✓ **Decision support evaluation** in controlled research settings  
✓ **Reproducibility studies** of published methods  

### 4. Out of Scope

✗ **Direct patient care** or clinical decision-making  
✗ **Unsupervised autonomous monitoring** without clinical oversight  
✗ **Therapeutic intervention** or closed-loop control  
✗ **Implantable device programming** or dosing decisions  

---

## What This Framework IS

### ✓ Signal Analysis and Feature Extraction

- Deterministic phase extraction via Hilbert transform
- Spectral, information-theoretic, and coupling feature computation
- Windowed analysis with transparent baseline normalization
- Triadic embedding for regime-change sensitivity

### ✓ Decision Support Evaluation

- Early-warning alert generation with configurable thresholds
- Risk stratification based on deviation metrics
- Lead-time and false-alarm rate characterization
- Performance metrics for prospective validation

### ✓ Research Validation Tool

- Preregistered analysis protocols
- Ablation studies (EEG-only, ECG-only, coupled)
- Synthetic benchmarking with ground truth
- Reproducible computational pipeline

### ✓ Educational Resource

- Demonstration notebooks with clear explanations
- Implementation of published signal processing methods
- Transparent operator definitions and equations
- Best practices for biosignal monitoring research

---

## What This Framework IS NOT

### ✗ NOT a Medical Device

- This software has **not been validated** for clinical use
- It has **not received regulatory approval** from FDA, CE, or any other authority
- It does **not replace** clinical judgment or medical expertise
- It is **not intended** for diagnosis, treatment, cure, or prevention of disease

### ✗ NOT a Therapeutic System

- Does **not provide** stimulation or neuromodulation
- Does **not control** implantable devices or actuators
- Does **not prescribe** medication or dosing
- Does **not perform** autonomous closed-loop intervention

### ✗ NOT a Biophysical Theory

- Does **not claim** that biological time follows new physical laws
- Does **not propose** fundamental theories of consciousness or physiology
- Uses established signal processing methods (Hilbert transform, entropy, coherence)
- Interpretive frameworks (ICE triangle) are heuristic, not mechanistic

### ✗ NOT Production-Ready Medical Software

- Has **not undergone** IEC 62304 software lifecycle validation
- Has **not completed** risk management per ISO 14971
- Has **not been tested** under IEC 60601 electrical safety standards
- Is **not compliant** with MDR (EU 2017/745) or equivalent regulations

---

## Safety Constraints

### 1. Human Subjects Research

**REQUIREMENT:** Any use with human subjects **must have**:

- ✓ Institutional Review Board (IRB) / Ethics Committee approval
- ✓ Written informed consent from all participants
- ✓ Appropriate data protection and privacy safeguards
- ✓ Clinical oversight by qualified medical personnel
- ✓ Preregistered study protocol with statistical analysis plan

**PROHIBITION:** Use on vulnerable populations (minors, prisoners, cognitively impaired) requires additional ethical review and may not be appropriate for research validation.

### 2. Clinical Oversight

**REQUIREMENT:** All monitoring sessions must include:

- ✓ Trained clinical personnel present during recording
- ✓ Standard-of-care monitoring in parallel (never as sole monitor)
- ✓ Emergency protocols independent of research system
- ✓ Ability to terminate study participation at any time

**PROHIBITION:** This system must **never** be the sole basis for clinical decisions.

### 3. Data Privacy and Security

**REQUIREMENT:**

- ✓ All biosignal data must be de-identified per HIPAA/GDPR
- ✓ Secure storage with encryption at rest and in transit
- ✓ Access controls and audit logging
- ✓ Data retention policies compliant with regulations

**PROHIBITION:** Do not use this system to process identifiable health information without appropriate data protection measures.

### 4. Failure Modes and Fallbacks

**KNOWN FAILURE MODES:**

- **Signal dropouts:** System cannot function with missing data
- **Artifact contamination:** May generate false alerts
- **Inter-subject variability:** Thresholds may not generalize
- **Medication effects:** May alter baseline characteristics
- **Comorbidities:** Not validated across diverse patient populations

**REQUIRED FALLBACKS:**

- ✓ Alert clinician to poor signal quality
- ✓ Default to "no decision" state under ambiguity
- ✓ Never suppress standard-of-care alarms or monitoring
- ✓ Log all decisions and alerts for post-hoc review

---

## Regulatory Considerations

### Medical Device Classification

If this methodology were to be translated into a medical device, it would likely be classified as:

- **FDA (United States):** Class II medical device (Software as a Medical Device)
- **EU (European Union):** Class IIa under MDR 2017/745
- **Japan (PMDA):** Class II medical device software

**These classifications require:**

- Clinical trials with prospective validation
- Risk management per ISO 14971
- Software lifecycle per IEC 62304
- Usability engineering per IEC 62366
- Electrical safety per IEC 60601 (for hardware components)
- Clinical evaluation and post-market surveillance

### Regulatory Pathways

**For future clinical translation, the following would be required:**

1. **Preclinical Validation:**
   - Performance benchmarking on annotated datasets
   - Failure mode and effects analysis (FMEA)
   - Software verification and validation (V&V)
   - Cybersecurity and data integrity assessment

2. **Clinical Investigation:**
   - Investigational Device Exemption (IDE) or equivalent
   - Prospective, preregistered clinical trials
   - Independent Data Safety Monitoring Board (DSMB)
   - Adverse event reporting protocols

3. **Regulatory Submission:**
   - 510(k) premarket notification (if predicate exists) or De Novo classification
   - CE marking technical file for EU market
   - Quality management system (ISO 13485)
   - Post-market surveillance and vigilance

**Current Status:** This framework has completed **none** of these regulatory requirements and is **not approved** for clinical use in any jurisdiction.

---

## Limitations and Known Issues

### 1. Technical Limitations

**Signal Processing:**
- Assumes stationarity within analysis windows (15-30 seconds)
- Hilbert transform edge effects at window boundaries
- Sensitivity to choice of baseline duration and window size
- Limited performance under high noise (SNR < 5 dB)

**Feature Extraction:**
- Permutation entropy sensitive to embedding dimension choice
- Phase synchronization assumes comparable frequency content
- Coherence requires sufficient spectral overlap
- LF/HF ratio interpretation dependent on recording conditions

**Detection Performance:**
- False alarm rates not characterized across diverse populations
- Lead-time distribution requires larger validation cohorts
- Threshold selection may need per-patient calibration
- No adaptive learning or online threshold optimization

### 2. Clinical Limitations

**Population Validity:**
- Validated only on synthetic and limited real datasets
- Not tested across age groups, comorbidities, medications
- Unknown performance in pediatric or geriatric populations
- Unclear generalization to different seizure types or cardiac conditions

**Recording Conditions:**
- Assumes controlled laboratory or clinical environment
- Not validated for ambulatory or home monitoring
- Sensitive to motion artifacts and electrode quality
- Requires simultaneous, synchronized EEG and ECG

**Clinical Utility:**
- Lead-time advantage over standard monitoring not yet established
- Clinical actions following alerts not defined or validated
- Cost-benefit analysis not performed
- Integration with clinical workflows not demonstrated

### 3. Software Limitations

**Implementation:**
- Research-grade code, not production-hardened
- Limited error handling for edge cases
- No real-time performance guarantees
- Dependencies on external libraries (scipy, numpy) not validated for medical use

**Testing:**
- Unit test coverage incomplete
- Integration testing limited to demonstration datasets
- No formal software verification per IEC 62304
- No cybersecurity penetration testing

**Scalability:**
- Not optimized for high-throughput processing
- Memory usage not characterized for long recordings
- Multi-channel processing scales linearly (not optimized)
- No distributed computing support

### 4. Methodological Limitations

**Statistical Power:**
- Synthetic validation provides proof-of-concept only
- Real-world validation cohorts needed for robust estimates
- Multiple comparison corrections not yet applied
- Publication bias toward positive results acknowledged

**Generalizability:**
- Framework designed for seizure and arrhythmia scenarios
- Unknown performance for other neurological/cardiac events
- Coupling assumptions may not hold for all conditions
- ICE interpretation heuristic, not mechanistically derived

---

## Ethical Considerations

### 1. Informed Consent

Research participants must be informed that:

- This is an **experimental monitoring system** under research validation
- It does **not replace standard clinical monitoring**
- Participation is **voluntary** and can be withdrawn at any time
- Data will be **de-identified and stored securely**
- Results may **not directly benefit** the participant (research study)
- **Risks include** potential for false alarms or missed events

### 2. Beneficence and Non-Maleficence

Researchers must:

- **Minimize risk** by maintaining parallel standard-of-care monitoring
- **Maximize benefit** by contributing to scientific knowledge
- **Avoid harm** by not using research alerts for clinical decisions
- **Report adverse events** through appropriate channels

### 3. Justice and Access

Considerations:

- Early-warning systems may not be equally accessible across populations
- Research should include diverse patient populations when safe
- Results should be published openly to benefit the field
- Technology transfer should consider global health equity

### 4. Data Ethics

Requirements:

- **Privacy:** De-identification and secure storage
- **Consent:** Explicit permission for data use and sharing
- **Secondary use:** Clear policies on data reuse
- **Ownership:** Transparent authorship and attribution

---

## Responsible Use Guidelines

### For Researchers

**DO:**
- ✓ Obtain appropriate ethics approval before human subjects research
- ✓ Preregister study protocols and analysis plans
- ✓ Report negative results and limitations transparently
- ✓ Share data and code for reproducibility (with appropriate protections)
- ✓ Acknowledge funding sources and conflicts of interest
- ✓ Cite original signal processing methods appropriately

**DO NOT:**
- ✗ Use for clinical decision-making without regulatory approval
- ✗ Make unsupported claims about clinical efficacy
- ✗ Deploy without appropriate clinical oversight
- ✗ Bypass ethics review for "low-risk" studies
- ✗ Cherry-pick results or selectively report findings

### For Clinicians

**DO:**
- ✓ Understand this is a research tool, not a medical device
- ✓ Maintain parallel standard-of-care monitoring at all times
- ✓ Report any adverse events or near-misses
- ✓ Provide feedback on usability and clinical relevance
- ✓ Participate in validation studies with appropriate protections

**DO NOT:**
- ✗ Rely on research system alerts for patient care decisions
- ✗ Use outside of approved research protocols
- ✗ Modify or reconfigure without documentation
- ✗ Deploy in unsupervised settings

### For Developers

**DO:**
- ✓ Follow software engineering best practices
- ✓ Document all modifications and parameter changes
- ✓ Conduct thorough testing before deployment
- ✓ Version control all code and configurations
- ✓ Report bugs and security vulnerabilities

**DO NOT:**
- ✗ Remove safety constraints or fail-safe mechanisms
- ✗ Deploy untested modifications in research studies
- ✗ Disable logging or audit trails
- ✗ Introduce adaptive/learning algorithms without validation

---

## Contact and Reporting

### Issue Reporting

**For bugs, errors, or technical issues:**
- GitHub Issues
- Label: `bug`, `documentation`, or `enhancement`
- Provide: version, dataset, configuration, error logs

**For security vulnerabilities:**
- Email: dfeen87@gmail.com
- Subject: "SECURITY: [brief description]"
- DO NOT disclose publicly until patch available

### Adverse Event Reporting

**If this system is used in a research study and an adverse event occurs:**

1. **Immediately report** to study IRB/Ethics Committee
2. **Document** the event thoroughly (timeline, context, outcome)
3. **Assess causality:** Was the research system a contributing factor?
4. **Notify** the research team and principal investigator
5. **File** per institutional adverse event reporting procedures

**Events requiring reporting:**
- Patient harm or near-miss related to system use
- False alarm leading to inappropriate clinical action
- Missed detection of clinically significant event
- Data breach or privacy violation
- Software malfunction affecting research integrity

### Scientific Inquiries

**For questions about methodology or collaboration:**
- Email: marcelkrueger092@gmail.com (Marcel Krüger)
- Email: dfeen87@gmail.com (Don Feeney)
- Subject: "Heart-Brain Monitoring Framework: [topic]"

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-27 | Initial release with comprehensive scope and safety documentation |

---

## License and Disclaimer

**License:** MIT

**DISCLAIMER:**

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

THIS SOFTWARE IS FOR RESEARCH USE ONLY AND HAS NOT BEEN APPROVED FOR CLINICAL USE BY ANY REGULATORY AUTHORITY. USE AT YOUR OWN RISK AND ONLY UNDER APPROPRIATE ETHICAL AND REGULATORY OVERSIGHT.

---

**END OF SCOPE AND SAFETY STATEMENT**
