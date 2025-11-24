# ============================================================================
# FILE 3: warm_start_v3.py - UPDATED
# ============================================================================
"""
Advanced Warm-Start with Knowledge Distillation
For RL integration
Usage: from warm_start_v3 import warm_start_v3
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import joblib

logger = logging.getLogger(__name__)

class PolicyNetworkV3(nn.Module):
    """Policy network for RL"""
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers=3, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class ValueNetworkV3(nn.Module):
    """Value network for RL"""
    def __init__(self, input_dim, hidden_dim, n_layers=3, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class AuxiliaryNetworkV3(nn.Module):
    """Auxiliary network for multi-task learning"""
    def __init__(self, input_dim, hidden_dim, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def warm_start_v3(policy_net, value_net, aux_net, X, y, teacher_model, 
                  cfg, device, epochs=20):
    """
    Advanced warm-start with knowledge distillation and multi-task learning
    
    Args:
        policy_net: Policy network
        value_net: Value network
        aux_net: Auxiliary network (optional)
        X: Training features (numpy array)
        y: Training labels (numpy array)
        teacher_model: Trained teacher model (sklearn) or None
        cfg: Configuration dict with 'batch_size', 'max_grad_norm'
        device: torch device (cpu or cuda)
        epochs: Number of warm-start epochs
    
    Returns:
        Trained networks (policy_net, value_net, aux_net)
    """
    
    logger.info(f"\n{'='*70}")
    logger.info("WARM-START V3 (Knowledge Distillation + Multi-Task)")
    logger.info(f"{'='*70}\n")
    
    # Validate config
    if "batch_size" not in cfg:
        cfg["batch_size"] = 64
    if "max_grad_norm" not in cfg:
        cfg["max_grad_norm"] = 0.3
    
    # Prepare tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    logger.info(f"Creating tensors: X={X_tensor.shape}, y={y_tensor.shape}")
    
    # Create teacher targets
    logger.info("Creating soft labels...")
    
    n_classes = len(np.unique(y))
    teacher_probs = None
    
    if teacher_model is not None:
        try:
            # Get soft labels from teacher model
            teacher_probs_np = teacher_model.predict_proba(X)
            teacher_probs = torch.from_numpy(teacher_probs_np).float()
            logger.info(f"✓ Using teacher model soft labels: {teacher_probs.shape}")
        except Exception as e:
            logger.warning(f"Could not get teacher predictions: {e}")
    
    # Fallback: synthetic soft labels
    if teacher_probs is None:
        logger.info("Creating synthetic soft labels (fallback)...")
        teacher_probs = torch.zeros(len(X), n_classes, dtype=torch.float32)
        
        for i in range(n_classes):
            mask = (y == i)
            teacher_probs[mask, i] = 0.9
            for j in range(n_classes):
                if j != i:
                    teacher_probs[mask, j] = 0.05
        
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
        logger.info(f"Created synthetic soft labels: {teacher_probs.shape}")
    
    # Create DataLoader
    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    
    logger.info(f"Created DataLoader: batch_size={cfg['batch_size']}")
    
    # Loss functions
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    ce_loss_fn = nn.CrossEntropyLoss()
    
    # Move to device
    policy_net = policy_net.to(device)
    value_net = value_net.to(device)
    if aux_net is not None:
        aux_net = aux_net.to(device)
    teacher_probs = teacher_probs.to(device)
    
    # Optimizers
    optimizer_policy = Adam(policy_net.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer_value = Adam(value_net.parameters(), lr=5e-4, weight_decay=1e-4)
    optimizer_aux = None
    if aux_net is not None:
        optimizer_aux = Adam(aux_net.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Training mode
    policy_net.train()
    value_net.train()
    if aux_net is not None:
        aux_net.train()
    
    logger.info(f"Starting warm-start training for {epochs} epochs...\n")
    
    for ep in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Get teacher targets for batch
            start_idx = batch_idx * cfg["batch_size"]
            end_idx = min(start_idx + cfg["batch_size"], len(X))
            teacher_batch = teacher_probs[start_idx:end_idx]
            
            # Policy warm-start (knowledge distillation)
            logits = policy_net(X_batch)
            student_log_probs = torch.log_softmax(logits / 2.0, dim=-1)
            teacher_probs_scaled = torch.softmax(teacher_batch / 2.0, dim=-1)
            kl_loss = kl_loss_fn(student_log_probs, teacher_probs_scaled)
            
            # Value warm-start
            values = value_net(X_batch).squeeze()
            value_loss = torch.mean((values - 0.5) ** 2)
            
            # Auxiliary task
            aux_loss = torch.tensor(0.0, device=device)
            if aux_net is not None:
                aux_logits = aux_net(X_batch)
                aux_loss = ce_loss_fn(aux_logits, y_batch)
            
            # Combined loss
            total_loss_batch = kl_loss + 0.5 * value_loss + 0.3 * aux_loss
            
            # Backpropagation
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            if optimizer_aux is not None:
                optimizer_aux.zero_grad()
            
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg["max_grad_norm"])
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), cfg["max_grad_norm"])
            if aux_net is not None:
                torch.nn.utils.clip_grad_norm_(aux_net.parameters(), cfg["max_grad_norm"])
            
            # Update
            optimizer_policy.step()
            optimizer_value.step()
            if optimizer_aux is not None:
                optimizer_aux.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(f"  Epoch {ep+1}/{epochs} | Avg Loss: {avg_loss:.6f}")
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ Warm-start complete!\n")
    
    return policy_net, value_net, aux_net