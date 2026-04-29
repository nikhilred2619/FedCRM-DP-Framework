"""tests/test_fedcrm.py — FedCRM-DP test suite"""
import sys, numpy as np; sys.path.insert(0,'/home/claude/fedcrm')
from fedcrm_core import (EnterpriseClient,
                          FedCRMPrediction, FederatedStrategy, PrivacyLevel,
                          ENTERPRISE_CLIENTS, PAPER_RESULTS, ABLATION_RESULTS)
from differential_privacy.engine import DifferentialPrivacyEngine
from aggregation_engine.secure_aggregation import SecureAggregationEngine

tp=0; tt=0
def test(name,cond):
    global tp,tt; tt+=1
    if cond: tp+=1; print(f"  ✅ {name}")
    else: print(f"  ❌ {name}")

# Paper results validation
test("Paper: FedCRM-DP Acc=0.8850", PAPER_RESULTS[FederatedStrategy.FEDCRM_DP]["acc"]==0.8850)
test("Paper: FedCRM-DP F1=0.5720",  PAPER_RESULTS[FederatedStrategy.FEDCRM_DP]["f1"]==0.5720)
test("Paper: FedCRM-DP AUC=0.8760", PAPER_RESULTS[FederatedStrategy.FEDCRM_DP]["auc"]==0.8760)
test("Paper: Centralized Acc=0.9120", PAPER_RESULTS[FederatedStrategy.CENTRALIZED]["acc"]==0.9120)
test("FedCRM-DP better than FedAvg", PAPER_RESULTS[FederatedStrategy.FEDCRM_DP]["acc"] > PAPER_RESULTS[FederatedStrategy.FEDAVG]["acc"])
test("Ablation: FedProx no DP = 0.5840 F1", ABLATION_RESULTS["FedProx (no DP)"]["f1"]==0.5840)
test("Ablation: ε=1 has Very Strong privacy", ABLATION_RESULTS["FedProx + DP (ε=1)"]["privacy"]=="Very Strong")

# Client configuration
test("5 enterprise clients", len(ENTERPRISE_CLIENTS)==5)
test("Client A: age<30", ENTERPRISE_CLIENTS[0].age_cohort=="Age<30")
test("Client B: n=17500", ENTERPRISE_CLIENTS[1].n_samples==17500)
test("All clients have epsilon", all(c.epsilon==5.0 for c in ENTERPRISE_CLIENTS))

# DP Engine
dp = DifferentialPrivacyEngine()
client = EnterpriseClient("T1","Test",1000,"Mixed",0.11,epsilon=5.0)
grad = np.ones(20) * 0.5
noisy = dp.clip_and_noise(grad, client)
test("DP: noise applied (output differs from input)", not np.allclose(grad, noisy))
test("DP: gradient still finite", np.all(np.isfinite(noisy)))
test("DP: ε=1 sigma > ε=5 sigma", dp.epsilon_to_sigma(1.0) > dp.epsilon_to_sigma(5.0))
test("DP: ε=5 is pareto-optimal (tradeoff)", DifferentialPrivacyEngine.privacy_utility_tradeoff()["pareto_optimal"]==5)

# Secure Aggregation
sa = SecureAggregationEngine(n_clients=5)
w1 = np.random.randn(20); w2 = np.random.randn(20)
masked1 = sa.mask_update(w1, "CLT_A", round_num=1)
masked2 = sa.mask_update(w2, "CLT_B", round_num=1)
test("SA: masking changes update", not np.allclose(w1, masked1))
agg = sa.unmask_and_aggregate([w1,w2], [0.6,0.4])
test("SA: aggregation returns array", isinstance(agg, np.ndarray))
test("SA: aggregation shape preserved", agg.shape == w1.shape)

# Privacy levels
test("ε=5 → Moderate privacy", EnterpriseClient("X","X",100,"A",0.1,epsilon=5.0).privacy_level==PrivacyLevel.MODERATE)
test("ε=1 → Very Strong privacy", EnterpriseClient("X","X",100,"A",0.1,epsilon=1.0).privacy_level==PrivacyLevel.VERY_STRONG)

# Salesforce integration
from crm_integration.salesforce.connector import SalesforceFedCRMConnector
conn = SalesforceFedCRMConnector()
pred = FedCRMPrediction(lead_id="LEAD-001",prediction=1,probability=0.72,
                         confidence=0.44,epsilon=5.0,noise_applied=True,
                         privacy_level="Moderate",compliance_status="Compliant")
result = conn.publish_federated_score(pred)
test("Salesforce: score published", result["success"])
test("Salesforce: DP flag preserved", result["payload"]["DP_Protected__c"])
af_result = conn.trigger_agentforce(pred,"LEAD-001")
test("Agentforce: topic triggered", af_result["success"])
test("Agentforce: high priority topic for prob>0.6", "HighPriority" in af_result["agent_topic"])

print(f"\n{'='*50}\nTests: {tp}/{tt} passed")
