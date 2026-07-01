"""
==============================================================================
  Amati Tiru Modifikasi (ATM) Module
  Provides behavioral cloning scaffolding to pre-train RL agents 
  using expert/heuristic trajectory data.
==============================================================================
"""
import os
import json
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .constants import HUMAN_DATA_DIR, STATE_DIM

class ATMDataloader(Dataset):
    """
    Dataloader for Behavioral Cloning (Imitation Learning).
    Reads (state, action) pairs recorded from expert demonstrations.
    """
    def __init__(self, data_dir=HUMAN_DATA_DIR):
        self.data_dir = data_dir
        self.states = []
        self.actions = []
        self.load_data()

    def load_data(self):
        """Load expert data from JSON logs."""
        if not os.path.exists(self.data_dir):
            return
            
        for file in os.listdir(self.data_dir):
            if file.endswith(".json"):
                path = os.path.join(self.data_dir, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        for step in data.get("steps", []):
                            # Pad state to match STATE_DIM if data is from older version
                            state = np.array(step["state"], dtype=np.float32)
                            if len(state) < STATE_DIM:
                                padded = np.zeros(STATE_DIM, dtype=np.float32)
                                padded[:len(state)] = state
                                state = padded
                            
                            self.states.append(state)
                            self.actions.append(np.array(step["action"], dtype=np.float32))
                except Exception as e:
                    print(f"[ATM] Failed to load {file}: {e}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return th.tensor(self.states[idx]), th.tensor(self.actions[idx])

def pretrain_policy(model, dataloader, epochs=5, lr=1e-4):
    """
    Pre-train the PPO policy network using Behavioral Cloning.
    """
    if len(dataloader) == 0:
        print("[ATM] No expert data found. Skipping pre-training.")
        return model

    print(f"[ATM] Starting Behavioral Cloning (Epochs: {epochs}, Samples: {len(dataloader.dataset)})")
    
    optimizer = th.optim.Adam(model.policy.parameters(), lr=lr)
    loss_fn = th.nn.MSELoss()
    
    model.policy.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            # PPO uses dict observation spaces for vision models
            # In ATM scaffolding, we're assuming purely state-based for simplicity,
            # or you'd need to mock the camera/lidar inputs as well.
            
            # This is a scaffold. Full implementation requires camera/lidar batching.
            pass
            
        print(f"[ATM] Epoch {epoch+1}/{epochs} Complete.")
        
    return model
