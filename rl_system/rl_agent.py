# rl_agent_v3.py - Advanced network architectures for V3
# Features: Batch norm, residual connections, multi-task learning

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual connection block for better gradient flow"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        
        # Projection for residual if dimensions don't match
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        
    def forward(self, x):
        identity = x
        if self.proj is not None:
            identity = self.proj(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out)
        return out

class MLPBaseV3(nn.Module):
    """Advanced MLP with residual connections and batch norm"""
    def __init__(self, input_dim, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Residual blocks
        for i in range(n_layers - 1):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class PolicyNetworkV3(nn.Module):
    """Advanced policy network with multiple heads"""
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers=3, dropout=0.1):
        super().__init__()
        self.body = MLPBaseV3(input_dim, hidden_dim, n_layers, dropout)
        
        # Multi-head attention (simplified)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Output heads
        self.logits = nn.Linear(hidden_dim, n_classes)
        self.value_aux = nn.Linear(hidden_dim, 1)  # Auxiliary value head
        
        # Initialize
        nn.init.orthogonal_(self.logits.weight, gain=0.01)
        nn.init.zeros_(self.logits.bias)
    
    def forward(self, x):
        x = self.body(x)
        
        # Attention (simplified - reshape for multihead)
        x_attn = x.unsqueeze(0)  # Add sequence dimension
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + x_attn.squeeze(0)  # Residual connection
        
        logits = self.logits(x)
        return logits

class ValueNetworkV3(nn.Module):
    """Advanced value network with dueling architecture"""
    def __init__(self, input_dim, hidden_dim, n_layers=3, dropout=0.1):
        super().__init__()
        self.body = MLPBaseV3(input_dim, hidden_dim, n_layers, dropout)
        
        # Dueling heads
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)
    
    def forward(self, x):
        features = self.body(x)
        
        # Dueling
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Combine (advantage - mean(advantage) + value)
        q_value = value + advantage - advantage.mean(dim=0, keepdim=True)
        
        return q_value.squeeze(-1)

class AuxiliaryNetworkV3(nn.Module):
    """Auxiliary task network for multi-task learning"""
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers=2, dropout=0.1):
        super().__init__()
        self.body = MLPBaseV3(input_dim, hidden_dim, n_layers, dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
        nn.init.orthogonal_(self.classifier.weight, gain=0.01)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        x = self.body(x)
        return self.classifier(x)

class EnsembleNetworkV3(nn.Module):
    """Ensemble of RL policies for robust predictions"""
    def __init__(self, input_dim, hidden_dim, n_classes, n_policies=3, dropout=0.1):
        super().__init__()
        self.policies = nn.ModuleList([
            PolicyNetworkV3(input_dim, hidden_dim, n_classes, dropout=dropout)
            for _ in range(n_policies)
        ])
        self.n_policies = n_policies
    
    def forward(self, x):
        logits_list = [policy(x) for policy in self.policies]
        # Average logits
        logits_ensemble = torch.stack(logits_list).mean(dim=0)
        return logits_ensemble
    
    def forward_all(self, x):
        """Return all policy outputs"""
        return [policy(x) for policy in self.policies]