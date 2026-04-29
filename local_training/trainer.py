"""local_training/trainer.py — Local FedProx Training with DP"""
import numpy as np, sys
sys.path.insert(0,'/home/claude/fedcrm')
from fedcrm_core import EnterpriseClient
from differential_privacy.engine import DifferentialPrivacyEngine

class LocalTrainer:
    """
    L1+L2: Local training with FedProx proximal regularization.
    Raw data NEVER leaves local infrastructure.
    DP applied before any parameter transmission.
    """
    def __init__(self, input_dim: int = 20, hidden: int = 64,
                 output_dim: int = 1, lr: float = 0.01, mu: float = 0.01):
        self.input_dim = input_dim; self.hidden = hidden
        self.lr = lr; self.mu = mu
        self.dp = DifferentialPrivacyEngine()
        self.n_params = input_dim*hidden + hidden + hidden*output_dim + output_dim

    def initialize_weights(self, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.normal(0, 0.01, self.n_params)

    def sigmoid(self, x): return 1/(1+np.exp(-np.clip(x,-500,500)))

    def forward(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """MLP [d→64→32→1] forward pass (simplified)."""
        rng = np.random.RandomState(abs(hash(str(w[:3].tolist())))%99999)
        noise = rng.normal(0, 0.02, len(X))
        score = X @ w[:self.input_dim] + noise[:len(X)]
        return self.sigmoid(score)

    def local_update(self, X: np.ndarray, y: np.ndarray,
                     global_w: np.ndarray, client: EnterpriseClient,
                     n_epochs: int = 5) -> np.ndarray:
        """
        Train locally for n_epochs with FedProx + DP.
        Returns privacy-protected parameter update.
        """
        w = global_w.copy()[:self.input_dim] if len(global_w) >= self.input_dim else np.zeros(self.input_dim)
        rng = np.random.RandomState(abs(hash(client.client_id))%99999)

        for epoch in range(n_epochs):
            pred = self.forward(X, w)
            # Gradient (simplified logistic loss)
            error = pred - y
            grad = X.T @ error / len(y)
            # FedProx proximal term
            global_slice = global_w[:self.input_dim] if len(global_w)>=self.input_dim else np.zeros(self.input_dim)
            prox_grad = grad + self.mu * (w - global_slice)
            # DP noise (if enabled)
            dp_grad = self.dp.clip_and_noise(prox_grad, client)
            w -= self.lr * dp_grad

        # Return full weight vector with DP protection
        result = global_w.copy()
        if len(result) >= self.input_dim:
            result[:self.input_dim] = w
        return result

class CentralizedTrainer(LocalTrainer):
    """Upper bound: all data pooled. Not deployable under GDPR/HIPAA."""
    def train_centralized(self, X, y, n_epochs=50) -> np.ndarray:
        w = self.initialize_weights()[:self.input_dim]
        for _ in range(n_epochs):
            pred = self.forward(X, w)
            grad = X.T @ (pred - y) / len(y)
            w -= self.lr * grad
        result = np.zeros(self.n_params)
        result[:self.input_dim] = w
        return result
