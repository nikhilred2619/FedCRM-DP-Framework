"""crm_integration/salesforce/connector.py — FedCRM-DP Salesforce Integration"""
import sys; sys.path.insert(0,'/home/claude/fedcrm')
from fedcrm_core import FedCRMPrediction
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SalesforceConfig:
    instance_url: str; access_token: str; api_version: str = "v60.0"

class SalesforceFedCRMConnector:
    """
    FedCRM-DP → Salesforce integration.
    Privacy-preserving lead scores published as Platform Events.
    No raw training data leaves enterprise boundaries.
    """
    SCORE_MAP = {(0,0.3):"Low",(0.3,0.6):"Medium",(0.6,1.0):"High"}

    def __init__(self, config: Optional[SalesforceConfig] = None):
        self.config = config; self._mock = config is None
        if self._mock: print("[SalesforceFedCRMConnector] Mock mode")

    def _lead_grade(self, prob):
        for (lo,hi),grade in self.SCORE_MAP.items():
            if lo <= prob < hi: return grade
        return "High"

    def publish_federated_score(self, pred: FedCRMPrediction) -> Dict[str,Any]:
        payload = {
            "FedCRM_Lead_ID__c":     pred.lead_id,
            "Conversion_Probability__c": pred.probability,
            "Lead_Grade__c":         self._lead_grade(pred.probability),
            "Privacy_Budget_ε__c":   pred.epsilon,
            "Privacy_Level__c":      pred.privacy_level,
            "DP_Protected__c":       pred.noise_applied,
            "Compliance_Status__c":  pred.compliance_status,
            "Federated_Model__c":    "FedCRM-DP_v1.0",
            "Audit_Events__c":       len(pred.audit_trail),
        }
        if self._mock:
            return {"success":True,"mock":True,"event":"FedCRM_Score__e","payload":payload}
        try:
            import requests
            url=f"{self.config.instance_url}/services/data/{self.config.api_version}/sobjects/FedCRM_Score__e/"
            r=requests.post(url,json=payload,headers={"Authorization":f"Bearer {self.config.access_token}","Content-Type":"application/json"})
            return {"success":r.status_code==201,"response":r.json()}
        except Exception as e:
            return {"success":False,"error":str(e)}

    def trigger_agentforce(self, pred: FedCRMPrediction, lead_id: str) -> Dict[str,Any]:
        topic = "FedCRM_HighPriority_Lead_Agent" if pred.probability > 0.6 else "FedCRM_Standard_Lead_Agent"
        ctx = {"lead_id":lead_id,"probability":pred.probability,"privacy_protected":True,"epsilon":pred.epsilon}
        if self._mock:
            return {"success":True,"mock":True,"agent_topic":topic,"context":ctx}
        try:
            import requests
            url=f"{self.config.instance_url}/services/data/{self.config.api_version}/einstein/copilot/actions/invoke"
            r=requests.post(url,json={"agentTopicName":topic,"context":ctx},headers={"Authorization":f"Bearer {self.config.access_token}","Content-Type":"application/json"})
            return {"success":r.status_code==200}
        except Exception as e:
            return {"success":False,"error":str(e)}
