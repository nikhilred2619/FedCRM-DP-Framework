# 🔵 FedCRM-DP Framework
## Privacy-Preserving Federated Learning for Secure Cross-Organization CRM Intelligence

> **"Collaborative CRM intelligence without sharing customer data"**

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Paper: IEEE Access](https://img.shields.io/badge/Paper-IEEE_Access-blue?style=for-the-badge)](https://orcid.org/0009-0006-7699-3928)

**Author:** Nikhil Reddy Donapati · Agentforce AI Specialist · Texas, USA

</div>

---

## The Problem: Data Isolation vs. Collaborative Intelligence

```
Bank A:      "Our fraud signals would improve Bank B's models — but we can't share customer data"
Hospital X:  "Patient risk patterns could help all providers — HIPAA prevents sharing"
Insurer Y:   "Cross-company churn patterns are valuable — GDPR forbids it"

Traditional Answer:  Train in isolation  → poor generalization
Centralized Answer:  Pool all data       → legally untenable
FedCRM-DP Answer:    Federated learning  → collaborative AI, no data sharing
```

---

## Validated Performance (UCI Bank Marketing, 100 rounds, MLP)

```
TABLE I: Comparative Performance — All Six Strategies

Strategy           Accuracy   F1-Score   AUC-ROC   Privacy Guarantee
─────────────────────────────────────────────────────────────────────
Local-Only         0.8140     0.3850     0.7420    None (isolated)
FedAvg             0.8750     0.5210     0.8540    None (gradient leakage)
FedProx            0.8890     0.5840     0.8810    None (gradient leakage)
FedAdam            0.8480     0.5718     0.9338    None (gradient leakage)
FedCRM-DP ε=5 ★   0.8850     0.5720     0.8760    (ε,δ)-DP + SecAgg ✓
Centralized †      0.9120     0.6480     0.9320    Not deployable (GDPR/HIPAA)

★ Proposed | † Upper bound only

FedCRM-DP retains 97.0% of FedProx F1 with formal privacy guarantees.
vs FedAvg: t=7.73, p=0.0015 | vs FedProx: t=−3.09, p=0.037
```

### Cross-Domain Validation (UCI Adult Income, N=48,842)
```
FedCRM-DP (ε=5): Acc=0.7840±0.0038, F1=0.6165±0.0022, AUC=0.8504
+1.79pp vs FedAvg (t=9.35, p=0.0112) | +1.74pp vs FedProx (t=9.13, p=0.0118)
```

---

## Framework Architecture

```
FedCRM-DP: Four-Layer Federated Stack

┌───────────────────────────────────────────────────────────────────┐
│  L1  LOCAL ENTERPRISE TRAINING                                    │
│       Raw data NEVER leaves organizational boundaries.            │
│       Five demographically partitioned clients (Bank Marketing):  │
│       Client A (age<30, n≈7.2k) | Client B (age30-45, n≈17.5k)  │
│       Client C (age46-60, n≈12.4k) | Client D (age>60, n≈5.1k)  │
│       Client E (mixed, n≈3.0k)                                    │
├───────────────────────────────────────────────────────────────────┤
│  L2  FEDPROX OPTIMIZATION                                         │
│       F_k(w) = f_k(w) + (μ/2)||w − w_t||²  (μ=0.01)             │
│       Bounds client drift under non-IID demographic distributions │
│       +6.3pp F1 over FedAvg in heterogeneous CRM settings        │
├───────────────────────────────────────────────────────────────────┤
│  L3  DIFFERENTIAL PRIVACY (DP-SGD)                                │
│       Per-sample gradient clipping: C=1.0                         │
│       Gaussian noise: N(0, σ²C²I), σ calibrated per ε             │
│       (ε,δ)-DP with δ=1e-5, ε∈{1,3,5,10}                        │
│       Pareto-optimal: ε=5 (97% F1 retention, moderate privacy)   │
├───────────────────────────────────────────────────────────────────┤
│  L4  SECURE AGGREGATION                                           │
│       Pairwise masking (Bonawitz et al. [5])                      │
│       Server learns ONLY weighted mean — not individual updates   │
│       w_{t+1} = Σ_k (n_k/n) · w_k^{t+1}                        │
│       Masks cancel in aggregate; server is blind to contributions │
└───────────────────────────────────────────────────────────────────┘
```

### Privacy-Utility Trade-off (Figure 3)
| ε | Accuracy | F1-Score | Privacy Level | Recommended For |
|:--:|:--:|:--:|:--:|---|
| 1 | 0.8440 | 0.4910 | Very Strong | Healthcare, regulated sectors |
| 3 | 0.8760 | 0.5520 | Strong | Financial services, GDPR-sensitive |
| **5 ★** | **0.8850** | **0.5720** | **Moderate** | **Standard CRM (Recommended)** |
| 10 | 0.8870 | 0.5800 | Minimal | Internal/low-sensitivity data |

---

## Quick Start

```bash
git clone https://github.com/nikhildonapati/fedcrm-dp.git
cd fedcrm-dp
pip install -r requirements.txt

# REST API
uvicorn api.main:app --reload --port 8000

# Run full evaluation (paper replication)
python3 evaluation/experiment.py

# Streamlit demo
streamlit run dashboard/app.py

# Docker
docker compose up --build
```

---

## REST API Example

```python
import requests
response = requests.post("http://localhost:8000/fedcrm/predict", json={
    "lead_id": "LEAD-001",
    "age": 35, "job_category": "management",
    "prior_contacts": 2, "campaign_duration": 250,
    "balance": 1500, "previous_outcome": "success",
    "epsilon": 5.0, "organization_id": "BANK-A"
})
# → {"prediction":1, "probability":0.724, "epsilon":5.0,
#    "privacy_level":"Moderate", "compliance_status":"Compliant",
#    "explanation":"FedCRM-DP: Likely to convert (72.4%)..."}
```

---

## Enterprise Use Cases

| Industry | Use Case | Privacy Concern | ε Setting |
|---|---|---|:--:|
| Banking | Lead conversion prediction | GDPR | 5 |
| Healthcare | Patient risk stratification | HIPAA | 1-3 |
| Insurance | Churn propensity | CCPA | 3-5 |
| Retail CRM | Customer LTV scoring | GDPR | 5 |
| SaaS | Account expansion prediction | SOC2 | 5 |
| HR Tech | Candidate success prediction | EEOC | 3 |

---

## Ablation Study (Table II)

| Configuration | Accuracy | F1 | Privacy |
|---|:---:|:---:|:---:|
| FedProx (no DP) | 0.8890 | 0.5840 | None |
| + DP (ε=10) | 0.8870 | 0.5800 | Minimal |
| **+ DP (ε=5) ★** | **0.8850** | **0.5720** | **Moderate** |
| + DP (ε=3) | 0.8760 | 0.5520 | Strong |
| + DP (ε=1) | 0.8440 | 0.4910 | Very Strong |
| **Full FedCRM-DP (ε=5)** | **0.8850** | **0.5720** | **Moderate+SA** |

---

## Project Structure

```
fedcrm-dp/
├── fedcrm_core.py                    ← Core data structures, paper results
├── federated_engine/orchestrator.py  ← Round orchestration, 100-round training
├── local_training/trainer.py         ← FedProx local training + DP
├── aggregation_engine/               ← Secure Aggregation (Bonawitz et al.)
├── differential_privacy/engine.py    ← DP-SGD: clipping + Gaussian noise
├── api/main.py                       ← FastAPI REST API (OpenAPI 3.0)
├── crm_integration/
│   ├── salesforce/connector.py       ← Salesforce + Agentforce
│   ├── servicenow/connector.py       ← ServiceNow workflows
│   ├── sap/connector.py              ← SAP customer ops
│   ├── oracle/connector.py           ← Oracle compliance
│   └── dynamics365/connector.py      ← MS Dynamics 365
├── evaluation/experiment.py          ← Full paper replication
├── dashboard/app.py                  ← Streamlit interactive demo
├── tests/test_fedcrm.py              ← 24/24 tests passing
├── results/fedcrm_results.json       ← Experimental results
├── Dockerfile + docker-compose.yml
└── requirements.txt
```

---

## Citation

```bibtex
@article{donapati2025fedcrmdp,
  title   = {FedCRM-DP: A Privacy-Preserving Federated Learning Framework 
             for Secure Cross-Organization CRM Intelligence},
  author  = {Donapati, Nikhil Reddy},
  journal = {IEEE Access},
  year    = {2025},
  note    = {Under Review},
  url     = {https://github.com/nikhildonapati/fedcrm-dp}
}
```

---

<div align="center">

**Nikhil Reddy Donapati · Texas, USA**  
*Agentforce AI Specialist & Senior Salesforce Developer*

</div>
