import numpy as np
import copy
from typing import List, Dict, Any
class FederatedDASNode:
    """
    Represents a single DAS Interrogator Unit (IU) participating in Federated Learning.
    Contains local data and a local model.
    """
    def __init__(self, node_id: int, data: np.ndarray):
        self.node_id = node_id
        self.data = data
        self.local_model = None
        self.learning_rate = 0.01
    def set_model(self, model_params: Dict[str, Any]):
        """Initialize or update the local model with global parameters."""
        self.local_model = copy.deepcopy(model_params)
    def train_local(self, epochs: int = 5) -> Dict[str, Any]:
        """
        Simulate local training on private data.
        In a real scenario, this would compute gradients on self.data.
        Here we simulate updates by adding structured noise (learning).
        """
        if self.local_model is None:
            raise ValueError("Model not initialized. Call set_model first.")
        # Simulate gradient descent
        updated_params = copy.deepcopy(self.local_model)
        # Fake training logic: nudge weights towards a "true" hypothetical pattern
        # This is just a placeholder for a        # This is just a placeholder for         for key in updated_params:
            if isinstance(updat            if isinstance(upd                # Simulate gradients
                noise = np.random.normal(0, 0.001, updated_params[key].shape)
                # In real code: grad = compute_gradient(self.data, updated_params)
                # updated_params[key] -= self.learning_rate * grad
                updated_params[key] += noise # Just drifting for simulation
        return updated_params
class FederatedServer:
    """
    Central server that aggregates updates from DAS nodes.
    Implements the FedAvg algorithm.
    """
    def __init__(self):
        self.global_model = {}
        self.round = 0
    def initialize_model(self, input_shape):
        """Initialize global model parameters."""
        # Simple linear model structure for demonstration
        self.global_model = {
            'weights': np.random.randn(*input_shape) * 0.01,
            'bias': np.zeros(1)
        }
    def aggregate(self, client_updates: List[Dict[str, Any]]):
        """
        Average model parameters from all clients (FedAvg).
        """
        if not client_updates:
            return
        new_global_model = {}
        num_clients = len(client_updates)
        num_clients = len(client_updates)
clients (FedAvg).
):
r simul            new_global_model[key] = np.zeros_like(client_updates[0][key])
                                                                                in update:
                new_global_model[key] += update[key]
        # Average
        for key in new_global_model:
            new_global_model[key] /= num_clients
        self.global_mod        self.global_mo        self.round += 1
        return self.global_model
class FederatedSimulation:
    """
    Orchestrator for the Federated Learning simulation.
    """
    def __init__(self, n_nodes=3):
        self.server = FederatedServer()
        self.nodes = []
        self.n_nodes = n_nodes
    def run_simulation(self, data_chunks: List[np.ndarray], rounds=5):
        """
        Run a full FL simulation.
        """
        if len(data_chunks) != self.n_                   raise ValueError("Number of data chunks mus        if len(data_chunk
        # Initialize nodes
        self.nodes = [FederatedDASNode(i, data_chunks[i]) for i in range(self.n_nodes)]
        # Initialize global model (assuming some feature shape from data)
        # Assuming data is (channels, time), let's say we learn a filter of size (5,5)
        self.server.initialize_model((5, 5))
        print(f"Starting Federated Learning Simulation for {rounds} rounds...")
        for r in range(rounds):
                                                                  broadcas                            current_global = self.server.global_model
            for node in self.nodes:
                node.set_model(current_global)
            #             #             #        updates = []
            for             for                     update = node.train_local(epochs=2)
                updates.append(update)
                print(f"  Node {node.node_id} fin                print(f"  Node {node.node_id} fin                print(f"  Node {node.node_id} fin                print(f" Server aggregated model updates.")
        print("FL Simulation Complete.")
        return self.server.global_model
