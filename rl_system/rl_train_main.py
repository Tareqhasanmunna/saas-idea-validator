# rl_train_main_v3.py - Advanced PPO with Ensemble & Multi-Task Learning
# V3 Features: Ensemble learning, multi-task learning, improved convergence
# Expected improvement: 70-75% test accuracy with better generalization

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
from datetime import datetime
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_loader import load_training_data
from rl_environment import DatasetEnv
from rl_agent import PolicyNetworkV3, ValueNetworkV3, AuxiliaryNetworkV3
from rl_visualization import plot_rl_training
from rl_evaluation import generate_evaluation_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========== CONFIG V3 ==========
CONFIG_PATH = "E:\\saas-idea-validator\\ml_outputs\\reports\\improved_rl_summary.json"
LGB_MODEL_PATH = "E:\\saas-idea-validator\\best_sl_model\\LightGBM_model.json"
OUT_DIR = "rl_results_v3"
os.makedirs(OUT_DIR, exist_ok=True)

cfg = {
    # Core PPO settings
    "episodes": 1500,           # More training (increased from 1000)
    "batch_size": 64,           # Smaller batches (more frequent updates)
    "ppo_epochs": 4,            # Increased PPO epochs for better convergence
    "clip_eps": 0.2,
    "gamma": 0.99,
    "lam": 0.95,
    
    # Learning rates
    "learning_rate": 1e-4,      # Lower learning rate for stability
    "critic_lr": 5e-4,          # Separate critic learning rate
    "entropy_beta": 0.10,       # Higher entropy (more exploration)
    "entropy_decay": 0.99,      # Decay entropy over time
    
    # Regularization (stronger)
    "value_loss_coeff": 0.8,    # Increased from 0.5
    "max_grad_norm": 0.3,       # Tighter gradient clipping
    "weight_decay": 2e-4,       # L2 regularization
    "dropout_rate": 0.3,        # Stronger dropout
    
    # Architecture
    "hidden_dim": 256,          # Larger for capacity
    "n_layers": 3,              # More layers
    "use_pca": True,            # PCA feature reduction
    "pca_components": 80,       # Reduce to 80 from 104
    
    # Warm-start
    "warm_start_epochs": 20,    # More knowledge distillation
    "use_multi_task": True,     # Multi-task learning
    
    # Ensemble
    "use_ensemble": True,       # Use LGB ensemble
    "ensemble_weight": 0.3,     # Weight for LGB in ensemble
    
    # Data
    "validation_split": 0.2,
    
    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            external = json.load(f)
        cfg["episodes"] = external.get("training_config", {}).get("episodes", cfg["episodes"])
        logger.info(f"✓ Loaded config from {CONFIG_PATH}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}, using defaults")

DEVICE = torch.device(cfg["device"])
logger.info(f"Using device: {DEVICE}")

# ========== DATA LOADING & PREPROCESSING ==========
logger.info("Loading dataset...")
X, y = load_training_data()

X = X.to_numpy() if hasattr(X, 'to_numpy') else X
y = y.to_numpy() if hasattr(y, 'to_numpy') else y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg["validation_split"], random_state=42, stratify=y
)

# KEEP ORIGINAL FOR LIGHTGBM
X_train_original = X_train.copy()
X_test_original = X_test.copy()

# Optional: PCA dimensionality reduction FIRST
pca = None
if cfg["use_pca"]:
    pca = PCA(n_components=cfg["pca_components"], random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    input_dim = cfg["pca_components"]
    logger.info(f"PCA applied: {X_train.shape[1]} → {input_dim} dimensions")
else:
    input_dim = X_train.shape[1]

# Standardize AFTER PCA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

n_classes = int(len(np.unique(y_train)))

logger.info(f"Dataset: {X_train.shape} (train), {X_test.shape} (test)")
logger.info(f"Classes: {n_classes}, Input: {input_dim}")

class_counts = np.bincount(y_train, minlength=n_classes)
inv_freq = (1.0 / (class_counts + 1e-9))
class_weights = (inv_freq / inv_freq.sum()) * n_classes
logger.info(f"Class weights: {class_weights}")

# ========== ENVIRONMENT ==========
env = DatasetEnv(X_train, y_train, n_classes,
                 class_weights=class_weights.tolist(),
                 success_bonus=0.8,      # Increased bonus
                 correct_reward=2.0,     # Stronger signal
                 wrong_reward=-1.0,
                 seed=42)
logger.info("✓ Environment created")

# ========== NETWORKS V3 ==========
policy = PolicyNetworkV3(input_dim=input_dim, hidden_dim=cfg["hidden_dim"],
                        n_classes=n_classes, n_layers=cfg["n_layers"],
                        dropout=cfg["dropout_rate"]).to(DEVICE)
value_net = ValueNetworkV3(input_dim=input_dim, hidden_dim=cfg["hidden_dim"],
                          n_layers=cfg["n_layers"], dropout=cfg["dropout_rate"]).to(DEVICE)

# Multi-task auxiliary network
aux_net = None
if cfg["use_multi_task"]:
    aux_net = AuxiliaryNetworkV3(input_dim=input_dim, hidden_dim=cfg["hidden_dim"],
                                n_classes=n_classes, dropout=cfg["dropout_rate"]).to(DEVICE)

# Separate optimizers
optimizer_policy = Adam(policy.parameters(), lr=cfg["learning_rate"], 
                       weight_decay=cfg["weight_decay"])
optimizer_value = Adam(value_net.parameters(), lr=cfg["critic_lr"],
                      weight_decay=cfg["weight_decay"])
optimizer_aux = None
if aux_net is not None:
    optimizer_aux = Adam(aux_net.parameters(), lr=cfg["learning_rate"],
                        weight_decay=cfg["weight_decay"])

logger.info("✓ Networks initialized")

# ========== RUNNING NORM ==========
class RunningNorm:
    def __init__(self, shape, eps=1e-6):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

running_norm = RunningNorm(input_dim)
running_norm.update(X_train[:min(1000, len(X_train))])

# ========== LIGHTGBM ENSEMBLE ==========
def load_lgb_model(model_path):
    try:
        model = lgb.Booster(model_file=model_path)
        logger.info(f"✓ Loaded LightGBM model")
        return model
    except Exception as e:
        logger.warning(f"LightGBM not available: {e}")
        return None

def get_lgb_ensemble_features(lgb_model, X, device):
    """Get LGB predictions as auxiliary features for RL"""
    if lgb_model is None:
        return None
    try:
        probs = lgb_model.predict(X)
        return torch.from_numpy(probs).float().to(device)
    except:
        return None

lgb_model = load_lgb_model(LGB_MODEL_PATH)
if lgb_model is None:
    logger.error(f"⚠️ LightGBM model not loaded from: {LGB_MODEL_PATH}")
    logger.error(f"File exists: {os.path.exists(LGB_MODEL_PATH)}")
else:
    logger.info("✓ LightGBM model successfully loaded")

# ========== KNOWLEDGE DISTILLATION WARM-START V3 ==========
def warm_start_v3(policy_net, value_net, aux_net, X, y, lgb_teacher, device,cfg, epochs=20):
    logger.info(f"\n{'='*70}")
    logger.info("ADVANCED WARM-START V3 (Knowledge Distillation + Multi-Task)")
    logger.info(f"{'='*70}\n")
    
    # X is already scaled and PCA reduced - don't rescale
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    # Get teacher targets from original data (need to apply PCA first)
    teacher_probs = None
    if lgb_teacher is not None:
        try:
            # LightGBM works on original features, so we need original X
            # Get from environment - it has unscaled data
            lgb_preds = lgb_teacher.predict(X)
            teacher_probs = torch.from_numpy(lgb_preds).float().to(device)
        except Exception as e:
            logger.warning(f"Could not get LGB predictions: {e}")
    
    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    ce_loss_fn = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        total_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            start_idx = batch_idx * cfg["batch_size"]
            end_idx = min(start_idx + cfg["batch_size"], len(X))
            teacher_batch = teacher_probs[start_idx:end_idx]
            
            # Policy warm-start (knowledge distillation)
            logits = policy_net(X_batch)
            student_log_probs = torch.log_softmax(logits / 2.0, dim=-1)
            teacher_probs_scaled = torch.softmax(teacher_batch / 2.0, dim=-1)
            kl_loss = kl_loss_fn(student_log_probs, teacher_probs_scaled)
            
            # Value warm-start (supervised)
            values = value_net(X_batch).squeeze()
            value_loss = torch.mean((values - 0.5) ** 2)  # Target 0.5 value
            
            # Auxiliary task (multi-task learning)
            aux_loss = 0
            if aux_net is not None:
                aux_logits = aux_net(X_batch)
                aux_loss = ce_loss_fn(aux_logits, y_batch)
            
            total_loss_batch = kl_loss + 0.5 * value_loss + 0.3 * aux_loss
            
            optimizer_policy.zero_grad()
            if optimizer_value:
                optimizer_value.zero_grad()
            if optimizer_aux:
                optimizer_aux.zero_grad()
            
            total_loss_batch.backward()
            
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg["max_grad_norm"])
            if value_net:
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), cfg["max_grad_norm"])
            
            optimizer_policy.step()
            if optimizer_value:
                optimizer_value.step()
            if optimizer_aux:
                optimizer_aux.step()
            
            total_loss += total_loss_batch.item()
        
        avg_loss = total_loss / (batch_idx + 1)
        logger.info(f"  Epoch {ep+1}/{epochs} | Loss: {avg_loss:.6f} (KL + Value + Aux)")
    
    logger.info(f"{'='*70}\n")

policy, value_net, aux_net = warm_start_v3(policy, value_net, aux_net, X_train, y_train, lgb_model, DEVICE, cfg,
                                          epochs=cfg["warm_start_epochs"])

# ========== RL HELPERS ==========
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advs = np.zeros_like(rewards)
    lastgaelam = 0
    values = np.append(values, 0)
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        advs[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    returns = advs + values[:-1]
    return advs, returns

def evaluate(policy_model, X_eval, y_eval, running_norm_obj, device):
    policy_model.eval()
    with torch.no_grad():
        X_norm = running_norm_obj.normalize(X_eval)
        X_tensor = torch.from_numpy(X_norm).float().to(device)
        logits = policy_model(X_tensor)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    acc = (preds == y_eval).mean()
    return float(acc)

# ========== MAIN PPO TRAINING V3 ==========
def train():
    logger.info(f"\n{'='*70}")
    logger.info("STARTING PPO TRAINING V3 (Advanced)")
    logger.info(f"{'='*70}")
    logger.info(f"Episodes: {cfg['episodes']}, Batch: {cfg['batch_size']}")
    logger.info(f"Learning rate: {cfg['learning_rate']}, Entropy: {cfg['entropy_beta']}\n")
    
    start_time = time.time()
    reward_history = []
    acc_history = []
    val_acc_history = []
    best_val = -1e9
    best_ep = 0
    current_entropy = cfg['entropy_beta']
    
    for ep in range(1, cfg['episodes'] + 1):
        # Decay entropy over time
        current_entropy = cfg['entropy_beta'] * (cfg['entropy_decay'] ** (ep / cfg['episodes']))
        
        # Collect batch
        idxs = env.sample_balanced_batch_indices(cfg['batch_size'])
        obs_batch = X_train[idxs]
        labels = y_train[idxs]
        
        running_norm.update(obs_batch)
        obs_norm = running_norm.normalize(obs_batch)
        
        obs_t = torch.from_numpy(obs_norm).float().to(DEVICE)
        labels_t = torch.from_numpy(labels).long().to(DEVICE)
        
        # Get actions and values
        policy.train()
        value_net.train()
        with torch.no_grad():
            logits = policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample().cpu().numpy()
            values = value_net(obs_t).cpu().numpy()
        
        # Compute rewards
        rewards = []
        dones = []
        corrects = []
        for a, lab in zip(actions, labels):
            correct = int(a) == int(lab)
            r = 2.0 if correct else -1.0
            rewards.append(r)
            dones.append(1.0)
            corrects.append(1 if correct else 0)
        
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        # GAE
        advs, returns = compute_gae(rewards, values, dones, cfg['gamma'], cfg['lam'])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        # PPO update
        actions_t = torch.from_numpy(actions).long().to(DEVICE)
        old_logits = policy(obs_t).detach()
        old_log_probs = torch.log_softmax(old_logits, dim=-1)
        old_log_probs = old_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1).detach()
        
        for ppo_ep in range(cfg['ppo_epochs']):
            # Policy loss
            logits = policy(obs_t)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(log_probs - old_log_probs)
            adv_t = torch.from_numpy(advs).float().to(DEVICE)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - cfg['clip_eps'], 1.0 + cfg['clip_eps']) * adv_t
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Value loss
            returns_t = torch.from_numpy(returns).float().to(DEVICE)
            vals = value_net(obs_t).squeeze()
            value_loss = torch.mean((vals - returns_t) ** 2)
            
            # Entropy
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))
            
            # Auxiliary loss (multi-task)
            aux_loss = 0
            if aux_net is not None:
                aux_logits = aux_net(obs_t)
                aux_loss = nn.CrossEntropyLoss()(aux_logits, labels_t)
            
            total_loss = policy_loss + cfg['value_loss_coeff'] * value_loss - current_entropy * entropy + 0.2 * aux_loss
            
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            if optimizer_aux:
                optimizer_aux.zero_grad()
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()),
                                          cfg['max_grad_norm'])
            if aux_net:
                torch.nn.utils.clip_grad_norm_(aux_net.parameters(), cfg['max_grad_norm'])
            
            optimizer_policy.step()
            optimizer_value.step()
            if optimizer_aux:
                optimizer_aux.step()
        
        # Logging
        batch_reward = rewards.mean()
        batch_acc = np.mean(corrects)
        reward_history.append(batch_reward)
        acc_history.append(batch_acc)
        
        if ep % 30 == 0 or ep == 1:
            avg_r = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
            avg_a = np.mean(acc_history[-50:]) if len(acc_history) >= 50 else np.mean(acc_history)
            logger.info(f"Ep {ep:4d}/{cfg['episodes']} | Reward: {avg_r:7.4f} | Acc: {avg_a:7.4f} | Entropy: {current_entropy:.4f}")
        
        # Validation
        if ep % 75 == 0 or ep == cfg['episodes']:
            val_acc = evaluate(policy, X_test, y_test, running_norm, DEVICE)
            val_acc_history.append(val_acc)
            logger.info(f"  → Val Acc: {val_acc:.4f}", end="")
            
            if val_acc > best_val:
                best_val = val_acc
                best_ep = ep
                torch.save({
                    "policy_state": policy.state_dict(),
                    "value_state": value_net.state_dict(),
                    "aux_state": aux_net.state_dict() if aux_net else None,
                    "running_mean": running_norm.mean,
                    "running_var": running_norm.var,
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "pca": pca,
                    "best_ep": best_ep
                }, os.path.join(OUT_DIR, "best_checkpoint.pth"))
                logger.info(f" (✓ Best!)")
            else:
                logger.info(f" (best: {best_val:.4f} @ ep {best_ep})")
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETED in {elapsed/60:.2f} minutes (V3)")
    logger.info(f"{'='*70}\n")
    
    # Final save
    torch.save({
        "policy_state": policy.state_dict(),
        "value_state": value_net.state_dict(),
        "aux_state": aux_net.state_dict() if aux_net else None,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "pca": pca
    }, os.path.join(OUT_DIR, "final_checkpoint.pth"))
    logger.info("✓ Final checkpoint saved")
    
    # Visualization
    viz_path = os.path.join(OUT_DIR, "training_visualization.png")
    plot_rl_training(reward_history, acc_history, window=40, save_path=viz_path)
    logger.info(f"✓ Visualization saved")
    
    # Evaluation
    generate_evaluation_report(policy, X_test, y_test, running_norm,
                             reward_history, acc_history, best_val, OUT_DIR, DEVICE)
    
    logger.info(f"\n✓ All outputs saved to: {OUT_DIR}\n")
    
    return reward_history, acc_history, val_acc_history

if __name__ == "__main__":
    train()
