import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FederatedUpdate:
    """A simple container for client updates."""

    params: Dict[str, np.ndarray]
    weight: float = 1.0


class FederatedDASNode:
    """Represents a DAS edge node participating in a toy federated-learning loop.

    Notes
    -----
    This module is intentionally lightweight and *does not* implement a full ML stack.
    It is used to demonstrate a reproducible FedAvg-style aggregation loop on
    NumPy arrays (e.g., model-parameter tensors).
    """

    def __init__(self, node_id: int, data: np.ndarray, learning_rate: float = 1e-2):
        self.node_id = int(node_id)
        self.data = data
        self.learning_rate = float(learning_rate)
        self.local_model: Optional[Dict[str, np.ndarray]] = None

    def set_model(self, model_params: Dict[str, np.ndarray]) -> None:
        """Set local model parameters from the global model."""

        self.local_model = {k: np.array(v, copy=True) for k, v in model_params.items()}

    def train_local(self, epochs: int = 1, seed: Optional[int] = None) -> FederatedUpdate:
        """Simulate a local training step.

        Performs gradient descent on a simple regression task:
        Minimize L(theta) = ||data - theta||^2 (estimation of mean).

        This mimics learning a site-specific background model.

        Returns
        -------
        FederatedUpdate
            Parameter dict and client weight.
        """

        if self.local_model is None:
            raise ValueError("Model not initialized. Call set_model() first.")

        # Task: Find the mean of the local data chunk.
        # Loss = 0.5 * sum((data - theta)^2)
        # Grad = -sum(data - theta) = -sum(data) + N*theta

        # We model 'theta' as a single vector matching the channel count (spatial mean)
        # or a scalar. For this demo, let's assume the model is a dictionary
        # mapping 'mean_trace' to a vector of shape (n_channels,).

        target_mean = np.mean(self.data, axis=1) # Target: temporal mean per channel (stationary background)

        # If parameters don't match data shape, just do the noise simulation (fallback)
        # This handles the generic 'model' case if the user initialized weirdly.

        params = copy.deepcopy(self.local_model)

        for name, theta in params.items():
            if theta.shape == target_mean.shape:
                # Perform GD
                for _ in range(epochs):
                    grad = (theta - target_mean) # simplified gradient (normalized by N)
                    theta -= self.learning_rate * grad
                params[name] = theta
            else:
                # Fallback for mismatched shapes: random walk (SGLD-ish)
                rng = np.random.default_rng(seed)
                noise = rng.normal(scale=1e-4, size=theta.shape)
                params[name] = theta + noise

        # Weight is proportional to dataset size (standard FedAvg)
        weight = float(self.data.size)

        return FederatedUpdate(params=params, weight=weight)


class FederatedServer:
    """Central server implementing FedAvg aggregation."""

    def __init__(self):
        self.global_model: Dict[str, np.ndarray] = {}
        self.round: int = 0

    def initialize_model(self, shape: tuple = (5, 5), seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        self.global_model = {
            "weights": rng.normal(loc=0.0, scale=1e-2, size=shape),
            "bias": np.zeros((1,), dtype=float),
        }
        self.round = 0
        return self.global_model

    def aggregate(self, updates: List[FederatedUpdate]) -> Dict[str, np.ndarray]:
        if not updates:
            raise ValueError("No client updates to aggregate")

        keys = list(updates[0].params.keys())
        for upd in updates[1:]:
            if list(upd.params.keys()) != keys:
                raise ValueError("All client updates must share the same parameter keys")

        total_w = float(sum(u.weight for u in updates))
        if total_w <= 0:
            total_w = float(len(updates))

        new_params: Dict[str, np.ndarray] = {}
        for k in keys:
            acc = np.zeros_like(updates[0].params[k], dtype=float)
            for u in updates:
                acc += (u.weight / total_w) * u.params[k]
            new_params[k] = acc

        self.global_model = new_params
        self.round += 1
        return self.global_model


class FederatedSimulation:
    """Orchestrates a small FedAvg simulation over NumPy data chunks."""

    def __init__(self, n_nodes: int = 3):
        self.n_nodes = int(n_nodes)
        self.server = FederatedServer()
        self.nodes: List[FederatedDASNode] = []

    def run_simulation(
        self,
        data_chunks: List[np.ndarray],
        rounds: int = 3,
        local_epochs: int = 1,
        model_shape: tuple = (5, 5),
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        if len(data_chunks) != self.n_nodes:
            raise ValueError(
                f"Expected {self.n_nodes} data chunks, got {len(data_chunks)}"
            )

        self.nodes = [FederatedDASNode(i, data_chunks[i]) for i in range(self.n_nodes)]
        self.server.initialize_model(shape=model_shape, seed=seed)

        for r in range(int(rounds)):
            current = self.server.global_model
            updates: List[FederatedUpdate] = []
            for node in self.nodes:
                node.set_model(current)
                updates.append(node.train_local(epochs=local_epochs, seed=seed + r + node.node_id))
            self.server.aggregate(updates)

        return self.server.global_model
