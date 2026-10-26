import os
import json
import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import numpy as np

from backend.models.gnn_models import build_model

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def load_mutag_dataset():
    dataset = TUDataset(root=os.path.join(DATA_DIR, "MUTAG"), name="MUTAG")
    return dataset

def get_splits(dataset, seed=42):
    labels = [data.y.item() for data in dataset]
    indices = np.arange(len(dataset))
    
    # Stratified 70% train, 15% val, 15% test
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    return train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx

def evaluate_model(model, loader, device="cpu"):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch.y.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            
    num_samples = len(all_targets)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs
    }

def train_single_model(
    model_name: str,
    epochs: int = 60,
    lr: float = 0.01,
    hidden_dim: int = 64,
    num_layers: int = 2,
    pooling: str = "mean",
    dropout: float = 0.2,
    batch_size: int = 16,
    seed: int = 42,
    save_checkpoint: bool = True
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset = load_mutag_dataset()
    train_data, val_data, test_data, _, _, _ = get_splits(dataset, seed=seed)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=model_name,
        in_dim=dataset.num_features,
        hidden_dim=hidden_dim,
        out_dim=dataset.num_classes,
        num_layers=num_layers,
        pooling=pooling,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }
    
    best_val_f1 = -1.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            preds = torch.argmax(out, dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
            
        train_loss = total_loss / total
        train_acc = correct / total
        
        val_metrics = evaluate_model(model, val_loader, device=device)
        
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_metrics["loss"], 4))
        history["val_acc"].append(round(val_metrics["accuracy"], 4))
        history["val_f1"].append(round(val_metrics["f1"], 4))
        
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    test_metrics = evaluate_model(model, test_loader, device=device)
    
    ckpt_path = os.path.join(WEIGHTS_DIR, f"{model_name.lower()}_best.pt")
    if save_checkpoint:
        torch.save({
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "config": {
                "in_dim": dataset.num_features,
                "hidden_dim": hidden_dim,
                "out_dim": dataset.num_classes,
                "num_layers": num_layers,
                "pooling": pooling,
                "dropout": dropout
            },
            "test_metrics": test_metrics,
            "history": history
        }, ckpt_path)
        
    return {
        "model_name": model_name,
        "config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "pooling": pooling,
            "dropout": dropout,
            "learning_rate": lr,
            "epochs": epochs
        },
        "history": history,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "auc": test_metrics["auc"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "loss": test_metrics["loss"]
        }
    }

def train_and_benchmark_all():
    models_to_run = [
        {"name": "GCN", "hidden_dim": 64, "num_layers": 2, "pooling": "mean", "lr": 0.005, "epochs": 50},
        {"name": "GraphSAGE", "hidden_dim": 64, "num_layers": 2, "pooling": "mean", "lr": 0.005, "epochs": 50},
        {"name": "GIN", "hidden_dim": 64, "num_layers": 2, "pooling": "sum", "lr": 0.005, "epochs": 50},
        {"name": "GAT", "hidden_dim": 64, "num_layers": 2, "pooling": "mean", "lr": 0.005, "epochs": 50},
    ]
    
    results = {}
    print("Beginning Training & Benchmark for all models on MUTAG...")
    for cfg in models_to_run:
        name = cfg["name"]
        print(f"--> Training {name}...")
        res = train_single_model(
            model_name=name,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            pooling=cfg["pooling"],
            dropout=0.2,
            seed=42,
            save_checkpoint=True
        )
        results[name] = res
        print(f"    Done {name}: Test Acc={res['test_metrics']['accuracy']:.4f}, Test F1={res['test_metrics']['f1']:.4f}")
        
    results_path = os.path.join(WEIGHTS_DIR, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results successfully written to {results_path}")
    return results

if __name__ == "__main__":
    train_and_benchmark_all()
