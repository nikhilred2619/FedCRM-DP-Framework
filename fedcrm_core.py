"""
fedcrm_core.py — FedCRM-DP Framework Core Data Structures

FedCRM-DP: Privacy-Preserving Federated Learning for Enterprise CRM
Four-layer architecture:
  L1 — Local Enterprise Training (raw data stays on-premise)
  L2 — FedProx Optimization (μ=0.01 proximal regularization)
  L3 — Differential Privacy (DP-SGD, ε=5 Pareto-optimal)
  L4 — Secure Aggregation (pairwise masking, Bonawitz et al. [5])

Validated Results (UCI Bank Marketing, 100 rounds, MLP):
  Local-Only:    Acc=0.8140, F1=0.3850, AUC=0.7420
  FedAvg:        Acc=0.8750, F1=0.5210, AUC=0.8540
  FedProx:       Acc=0.8890, F1=0.5840, AUC=0.8810
  FedAdam:       Acc=0.8480, F1=0.5718, AUC=0.9338
  FedCRM-DP ε=5: Acc=0.8850, F1=0.5720, AUC=0.8760 ← Proposed
  Centralized†:  Acc=0.9120, F1=0.6480, AUC=0.9320  (not deployable)

Pareto-optimal: ε=5 retains 97.0% of FedProx F1 with formal privacy guarantees.

Reference: Donapati, N.R. (2025). FedCRM-DP: A Privacy-Preserving Federated
           Learning Framework. IEEE Access (Under Review).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time, uuid

class FederatedStrategy(str, Enum):
    LOCAL_ONLY  = "Local-Only"
    FEDAVG      = "FedAvg"
    FEDPROX     = "FedProx"
    FEDADAM     = "FedAdam"
    FEDCRM_DP   = "FedCRM-DP"
    CENTRALIZED = "Centralized"

class PrivacyLevel(str, Enum):
    NONE        = "None"
    MINIMAL     = "Minimal"    # ε=10
    MODERATE    = "Moderate"   # ε=5  ← Recommended
    STRONG      = "Strong"     # ε=3
    VERY_STRONG = "VeryStrong" # ε=1

@dataclass
class EnterpriseClient:
    """
    Represents one federated enterprise client.
    Each client holds local CRM data that NEVER leaves its infrastructure.
    """
    client_id:      str
    org_name:       str
    n_samples:      int
    age_cohort:     str          # Demographic partition
    positive_rate:  float        # Class imbalance local to this org
    domain:         str = "Banking"

    # Privacy settings
    epsilon:        float = 5.0  # DP budget ε (Pareto-optimal)
    delta:          float = 1e-5 # DP failure probability δ
    clip_norm:      float = 1.0  # Gradient clipping norm C

    # Local model weights (initialized to global)
    local_weights:  Optional[np.ndarray] = None

    # Per-round metrics
    round_metrics:  List[Dict] = field(default_factory=list)

    @property
    def noise_multiplier(self) -> float:
        """σ calibrated from ε via moments accountant (simplified)."""
        epsilon_to_sigma = {1.0: 1.10, 3.0: 0.70, 5.0: 0.55, 10.0: 0.38}
        return epsilon_to_sigma.get(self.epsilon, 0.55)

    @property
    def privacy_level(self) -> PrivacyLevel:
        if self.epsilon <= 1:   return PrivacyLevel.VERY_STRONG
        if self.epsilon <= 3:   return PrivacyLevel.STRONG
        if self.epsilon <= 5:   return PrivacyLevel.MODERATE
        if self.epsilon <= 10:  return PrivacyLevel.MINIMAL
        return PrivacyLevel.NONE

    @property
    def data_weight(self) -> float:
        """Federated averaging weight: nk/n."""
        return self.n_samples / 45211  # Bank Marketing total

@dataclass
class FederatedRoundResult:
    """Results from one communication round."""
    round_num:       int
    strategy:        FederatedStrategy
    global_accuracy: float
    global_f1:       float
    global_auc:      float
    epsilon:         float
    privacy_level:   PrivacyLevel
    n_clients:       int = 5
    timestamp:       float = field(default_factory=time.time)

@dataclass
class FedCRMPrediction:
    """Output of FedCRM-DP lead conversion prediction."""
    prediction_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    lead_id:         str = ""

    # Prediction
    prediction:      int = 0        # 0=no conversion, 1=conversion
    probability:     float = 0.0    # Conversion probability
    confidence:      float = 0.0    # Model confidence

    # Privacy
    epsilon:         float = 5.0
    noise_applied:   bool = True
    privacy_level:   str = "Moderate"

    # Governance
    compliance_status: str = "Compliant"
    audit_trail:     List[str] = field(default_factory=list)

    # Performance
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id":    self.prediction_id,
            "lead_id":          self.lead_id,
            "prediction":       self.prediction,
            "probability":      round(self.probability, 3),
            "confidence":       round(self.confidence, 3),
            "epsilon":          self.epsilon,
            "privacy_level":    self.privacy_level,
            "compliance_status": self.compliance_status,
            "audit_events":     len(self.audit_trail),
            "inference_time_ms": round(self.inference_time_ms, 2),
        }

# Enterprise client configuration (Table in paper)
ENTERPRISE_CLIENTS = [
    EnterpriseClient("CLT_A","BankCorp-West",  7200,  "Age<30",  0.07),
    EnterpriseClient("CLT_B","FinServ-Central",17500, "Age30-45",0.12),
    EnterpriseClient("CLT_C","InsureCo-East",  12400, "Age46-60",0.13),
    EnterpriseClient("CLT_D","Wealth-Partners",5100,  "Age>60",  0.09),
    EnterpriseClient("CLT_E","CRM-Solutions",  3011,  "Mixed",   0.11),
]

# Paper Table I results
PAPER_RESULTS = {
    FederatedStrategy.LOCAL_ONLY:  {"acc":0.8140,"f1":0.3850,"auc":0.7420},
    FederatedStrategy.CENTRALIZED: {"acc":0.9120,"f1":0.6480,"auc":0.9320},
    FederatedStrategy.FEDAVG:      {"acc":0.8750,"f1":0.5210,"auc":0.8540},
    FederatedStrategy.FEDPROX:     {"acc":0.8890,"f1":0.5840,"auc":0.8810},
    FederatedStrategy.FEDADAM:     {"acc":0.8480,"f1":0.5718,"auc":0.9338},
    FederatedStrategy.FEDCRM_DP:   {"acc":0.8850,"f1":0.5720,"auc":0.8760},
}

# Paper Table II ablation
ABLATION_RESULTS = {
    "FedProx (no DP)":         {"acc":0.8890,"f1":0.5840,"auc":0.8810,"privacy":"None"},
    "FedProx + DP (ε=10)":     {"acc":0.8870,"f1":0.5800,"auc":0.8780,"privacy":"Minimal"},
    "FedProx + DP (ε=5)":      {"acc":0.8850,"f1":0.5720,"auc":0.8760,"privacy":"Moderate"},
    "FedProx + DP (ε=3)":      {"acc":0.8760,"f1":0.5520,"auc":0.8640,"privacy":"Strong"},
    "FedProx + DP (ε=1)":      {"acc":0.8440,"f1":0.4910,"auc":0.8220,"privacy":"Very Strong"},
    "FedCRM-DP Full (ε=5) ★":  {"acc":0.8850,"f1":0.5720,"auc":0.8760,"privacy":"Moderate+SA"},
}
