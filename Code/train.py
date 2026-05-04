"""
Multi-View Attention Multiple-Instance Learning for Cognitive Distortion Detection
ACL 2026 Main
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gc
import pickle
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ── Hyperparameters 
HIDDEN_DIM   = 128
DROPOUT_RATE = 0.3
INITIAL_LR   = 0.0005
MIN_LR       = 0.00001
DECAY_RATE   = 0.00001
NUM_EPOCHS   = 100
NUM_RUNS     = 10
PATIENCE     = 10
BATCH_SIZE   = 32
NUM_CLASSES  = 10
NUM_HEADS    = 4
INPUT_DIM    = 384


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiViewGatedAttentionMIL(nn.Module):
    """
    Multi-View Gated Attention MIL model (Section 6.3 in the paper).
    Integrates LLM-inferred instance representations and salience scores
    with original sentence embeddings via multi-view gated attention.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=NUM_CLASSES, dropout_rate=DROPOUT_RATE, num_heads=NUM_HEADS):
        super().__init__()
        self.num_heads = num_heads

        # Instance-level feature and gate networks (one per attention view)
        self.feature_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
            for _ in range(num_heads)
        ])
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])

        # Original sentence feature network
        self.original_feat_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        # Fusion and classifier
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * (num_heads + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, prob, original_emb):
        """
        Args:
            x           : instance embeddings  (batch, max_instances, input_dim)
            prob        : salience scores       (batch, max_instances)
            original_emb: sentence embeddings  (batch, input_dim)
        """
        head_outputs = []
        for i in range(self.num_heads):
            # Gated attention weighted by LLM-assigned salience scores (Eq. 3)
            gate_scores    = self.gate_nets[i](x).squeeze(-1)           # (batch, max_instances)
            features       = self.feature_nets[i](x)                    # (batch, max_instances, hidden)
            gated_features = gate_scores.unsqueeze(-1) * features * prob.unsqueeze(-1)
            bag_repr       = torch.sum(gated_features, dim=1)           # (batch, hidden)
            head_outputs.append(bag_repr)

        # Multi-view aggregation (Eq. 4) — average across views via concat + linear
        original_feat = self.original_feat_net(original_emb)            # (batch, hidden)
        fused         = torch.cat(head_outputs + [original_feat], dim=1)
        fused_output  = self.fusion_net(fused)
        return self.classifier(fused_output)


# ─────────────────────────────────────────────────────────────────────────────
# Data split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(labels, val_ratio=0.1, test_ratio=0.1):
    """Stratified train/val/test split by class (Section 6.4 in the paper)."""
    val_indices, test_indices = [], []
    for cls in range(NUM_CLASSES):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        val_n  = int(len(cls_indices) * val_ratio)
        test_n = int(len(cls_indices) * test_ratio)
        val_indices.extend(cls_indices[:val_n])
        test_indices.extend(cls_indices[val_n:val_n + test_n])

    val_indices   = list(set(val_indices))
    test_indices  = sorted(list(set(test_indices) - set(val_indices)))
    train_indices = sorted(list(set(range(len(labels))) - set(val_indices) - set(test_indices)))
    return train_indices, val_indices, test_indices


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(instance_emb, probabilities, original_emb, labels, seed):
    """Single training run with the given random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n  Seed: {seed}")

    # Data split
    label_np = labels.numpy()
    train_idx, val_idx, test_idx = stratified_split(label_np)

    dataset = torch.utils.data.TensorDataset(
        instance_emb, probabilities, original_emb, labels
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False
    )

    # Model, optimizer, loss
    model     = MultiViewGatedAttentionMIL()
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss()
    current_lr = INITIAL_LR

    best_val_f1      = 0.0
    best_model_state = None
    best_val_metrics = {}
    patience_counter = 0
    min_val_loss     = float("inf")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for x_batch, prob_batch, orig_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch, prob_batch, orig_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Linear learning rate decay (Section 6.4)
        current_lr = max(current_lr - DECAY_RATE, MIN_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Validation
        model.eval()
        val_true, val_pred, val_loss_total = [], [], 0.0
        with torch.no_grad():
            for x_batch, prob_batch, orig_batch, y_batch in val_loader:
                output       = model(x_batch, prob_batch, orig_batch)
                val_loss_total += criterion(output, y_batch).item()
                val_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss   = val_loss_total / len(val_loader)
        print(f"    Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f}")

        # Early stopping
        if avg_val_loss < min_val_loss:
            min_val_loss     = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}.")
                break

        # Save best model by validation F1
        _, _, val_f1, _ = precision_recall_fscore_support(
            val_true, val_pred, average="weighted", zero_division=0
        )
        val_acc                  = accuracy_score(val_true, val_pred)
        val_prec_w, val_rec_w, _, _ = precision_recall_fscore_support(
            val_true, val_pred, average="weighted", zero_division=0
        )
        val_prec_m, val_rec_m, val_f1_m, _ = precision_recall_fscore_support(
            val_true, val_pred, average="macro", zero_division=0
        )

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_metrics = {
                "acc": val_acc,
                "prec_w": val_prec_w, "rec_w": val_rec_w, "f1_w": val_f1,
                "prec_m": val_prec_m, "rec_m": val_rec_m, "f1_m": val_f1_m,
            }

    # Test evaluation with best model
    model.load_state_dict(best_model_state)
    model.eval()
    test_true, test_pred, test_probs = [], [], []
    with torch.no_grad():
        for x_batch, prob_batch, orig_batch, y_batch in test_loader:
            output = model(x_batch, prob_batch, orig_batch)
            probs  = torch.softmax(output, dim=1)
            test_probs.extend(probs.cpu().numpy())
            test_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            test_true.extend(y_batch.cpu().numpy())

    acc               = accuracy_score(test_true, test_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        test_true, test_pred, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        test_true, test_pred, average="macro", zero_division=0
    )
    class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(
        test_true, test_pred, average=None, zero_division=0
    )

    test_metrics = {
        "acc": acc,
        "prec_w": prec_w, "rec_w": rec_w, "f1_w": f1_w,
        "prec_m": prec_m, "rec_m": rec_m, "f1_m": f1_m,
        "class_prec": class_prec, "class_rec": class_rec, "class_f1": class_f1,
        "test_indices": test_idx,
        "true_labels": test_true,
        "predicted_labels": test_pred,
        "prediction_probs": test_probs,
    }

    return best_val_metrics, test_metrics, best_model_state


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(mode, npz_path, output_dir="."):
    print(f"\n{'='*60}")
    print(f"Dataset: {mode}  |  File: {npz_path}")
    print(f"{'='*60}")

    # Load data
    data               = np.load(npz_path, allow_pickle=True)
    embeddings         = torch.tensor(data["embeddings"], dtype=torch.float32)
    instance_emb       = embeddings[:, 2:, :]   # instance embeddings (index 2 onward)
    original_emb       = embeddings[:, 0, :]    # original sentence embedding (index 0)
    labels             = torch.tensor(np.argmax(data["labels"], axis=1), dtype=torch.long)
    probabilities      = torch.tensor(data["probabilities"], dtype=torch.float32)

    # Generate random seeds for each run
    seeds = random.sample(range(1000), NUM_RUNS)
    print(f"Random seeds for this experiment: {seeds}")

    # Accumulators
    val_metrics_all  = []
    test_metrics_all = []

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
        val_metrics, test_metrics, model_state = train_one_run(
            instance_emb, probabilities, original_emb, labels, seed=seeds[run]
        )
        val_metrics_all.append(val_metrics)
        test_metrics_all.append(test_metrics)

        # Save model checkpoint
        torch.save(model_state, f"{output_dir}/model_{mode}_run{run+1}.pt")

        # Save per-run test results as CSV
        results_df = pd.DataFrame({
            "data_index":    test_metrics["test_indices"],
            "true_label":    test_metrics["true_labels"],
            "predicted_label": test_metrics["predicted_labels"],
            "correct": [
                1 if t == p else 0
                for t, p in zip(test_metrics["true_labels"], test_metrics["predicted_labels"])
            ],
            "confidence": [max(prob) for prob in test_metrics["prediction_probs"]],
            **{
                f"prob_class_{c}": [prob[c] for prob in test_metrics["prediction_probs"]]
                for c in range(NUM_CLASSES)
            },
        })
        results_df.to_csv(f"{output_dir}/test_results_{mode}_run{run+1}.csv", index=False)

    # ── Summary statistics
    def mean_std(key):
        vals = [m[key] for m in test_metrics_all]
        return np.mean(vals), np.std(vals)

    print(f"\n{'='*60}")
    print(f"Summary — {mode}  ({NUM_RUNS} runs)")
    print(f"{'='*60}")
    print("[ Test — Weighted ]")
    print(f"  Accuracy : {mean_std('acc')[0]:.4f} ± {mean_std('acc')[1]:.4f}")
    print(f"  Precision: {mean_std('prec_w')[0]:.4f} ± {mean_std('prec_w')[1]:.4f}")
    print(f"  Recall   : {mean_std('rec_w')[0]:.4f} ± {mean_std('rec_w')[1]:.4f}")
    print(f"  F1       : {mean_std('f1_w')[0]:.4f} ± {mean_std('f1_w')[1]:.4f}")
    print("[ Test — Macro ]")
    print(f"  Precision: {mean_std('prec_m')[0]:.4f} ± {mean_std('prec_m')[1]:.4f}")
    print(f"  Recall   : {mean_std('rec_m')[0]:.4f} ± {mean_std('rec_m')[1]:.4f}")
    print(f"  F1       : {mean_std('f1_m')[0]:.4f} ± {mean_std('f1_m')[1]:.4f}")

    print("\n[ Per-class F1 ]")
    for c in range(NUM_CLASSES):
        f1_vals = [m["class_f1"][c] for m in test_metrics_all]
        print(f"  Class {c}: {np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}")

    print("[ Validation — Weighted ]")
    print(f"  F1: {np.mean([m['f1_w'] for m in val_metrics_all]):.4f} "
          f"± {np.std([m['f1_w'] for m in val_metrics_all]):.4f}")

    # Save combined results
    all_runs = []
    for run, m in enumerate(test_metrics_all):
        df = pd.read_csv(f"{output_dir}/test_results_{mode}_run{run+1}.csv")
        df["run"] = run + 1
        all_runs.append(df)
    combined = pd.concat(all_runs, ignore_index=True)
    combined.to_csv(f"{output_dir}/test_results_{mode}_all_runs.csv", index=False)

    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multi-View Gated Attention MIL for cognitive distortion detection"
    )
    parser.add_argument("--elb",    required=True, help="Path to ELB-processed .npz file")
    parser.add_argument("--no_elb", required=True, help="Path to non-ELB-processed .npz file")
    parser.add_argument("--output_dir", default=".", help="Directory to save results (default: .)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    file_paths = {
        "ELB":    args.elb,
        "no_ELB": args.no_elb,
    }

    for mode, path in file_paths.items():
        run_experiment(mode, path, output_dir=args.output_dir)
