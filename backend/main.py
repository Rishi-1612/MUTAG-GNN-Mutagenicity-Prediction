import os
import json
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend.models.gnn_models import build_model
from backend.train_eval import (
    load_mutag_dataset,
    train_single_model,
    WEIGHTS_DIR,
    DATA_DIR
)
from backend.molecule_utils import (
    mutag_to_json_graph,
    smiles_to_graph_and_pyg,
    PRESET_MOLECULES,
    ATOM_SYMBOLS,
    ATOM_NAMES,
    ATOM_COLORS,
    BOND_TYPES
)
from backend.explainability import compute_node_importance
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="MUTAG GNN Mutagenicity Predictor & Analysis API",
    description="Backend API for Graph Neural Network molecular classification and explainability on the MUTAG benchmark.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for dataset and loaded models
MUTAG_DATASET = None
LOADED_MODELS = {}

def get_dataset():
    global MUTAG_DATASET
    if MUTAG_DATASET is None:
        MUTAG_DATASET = load_mutag_dataset()
    return MUTAG_DATASET

def load_cached_model(model_name: str):
    name = model_name.upper()
    if name in LOADED_MODELS:
        return LOADED_MODELS[name]
        
    ckpt_file = os.path.join(WEIGHTS_DIR, f"{name.lower()}_best.pt")
    if not os.path.exists(ckpt_file):
        # If model weight does not exist yet, train single model quickly
        print(f"Weight not found for {name}. Running quick training...")
        train_single_model(model_name=name, epochs=30, lr=0.005)
        
    ckpt = torch.save if False else torch.load(ckpt_file, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(
        model_name=name,
        in_dim=cfg.get("in_dim", 7),
        hidden_dim=cfg.get("hidden_dim", 64),
        out_dim=cfg.get("out_dim", 2),
        num_layers=cfg.get("num_layers", 2),
        pooling=cfg.get("pooling", "mean"),
        dropout=cfg.get("dropout", 0.2)
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    LOADED_MODELS[name] = {
        "model": model,
        "config": cfg,
        "test_metrics": ckpt.get("test_metrics", {})
    }
    return LOADED_MODELS[name]

@app.on_event("startup")
async def startup_event():
    print("Pre-loading MUTAG dataset and models...")
    get_dataset()
    for m in ["GIN", "GCN", "GRAPHSAGE", "GAT"]:
        try:
            load_cached_model(m)
        except Exception as e:
            print(f"Failed to pre-cache {m}: {e}")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "dataset_loaded": MUTAG_DATASET is not None
    }

@app.get("/api/dataset/overview")
def get_dataset_overview():
    ds = get_dataset()
    num_graphs = len(ds)
    
    label_counts = {0: 0, 1: 0}
    total_nodes = 0
    total_edges = 0
    node_counts_list = []
    edge_counts_list = []
    atom_counts = {s: 0 for s in ATOM_SYMBOLS}
    bond_counts = {b: 0 for b in BOND_TYPES}
    
    for i in range(num_graphs):
        data = ds[i]
        lbl = int(data.y.item())
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
        n_nodes = data.x.size(0)
        n_edges = data.edge_index.size(1) // 2  # undirected
        total_nodes += n_nodes
        total_edges += n_edges
        node_counts_list.append(n_nodes)
        edge_counts_list.append(n_edges)
        
        # Atom counts
        feat_sums = data.x.sum(dim=0).cpu().numpy()
        for idx, s in enumerate(ATOM_SYMBOLS):
            atom_counts[s] += int(feat_sums[idx])
            
        # Bond counts
        if data.edge_attr is not None:
            # sum over edges divided by 2 (undirected)
            b_sums = (data.edge_attr.sum(dim=0).cpu().numpy() / 2.0).round().astype(int)
            for idx, b in enumerate(BOND_TYPES):
                bond_counts[b] += int(b_sums[idx])
                
    return {
        "total_graphs": num_graphs,
        "num_features": ds.num_features,
        "num_classes": ds.num_classes,
        "class_distribution": {
            "mutagenic": label_counts.get(1, 0),
            "non_mutagenic": label_counts.get(0, 0),
            "mutagenic_ratio": round(label_counts.get(1, 0) / num_graphs, 3)
        },
        "graph_statistics": {
            "avg_nodes": round(total_nodes / num_graphs, 2),
            "min_nodes": int(min(node_counts_list)),
            "max_nodes": int(max(node_counts_list)),
            "avg_edges": round(total_edges / num_graphs, 2),
            "min_edges": int(min(edge_counts_list)),
            "max_edges": int(max(edge_counts_list))
        },
        "atom_frequencies": atom_counts,
        "bond_frequencies": bond_counts,
        "atom_metadata": {
            s: {
                "name": ATOM_NAMES.get(s, s),
                "color": ATOM_COLORS.get(s, "#64748B"),
                "count": atom_counts[s]
            }
            for s in ATOM_SYMBOLS
        }
    }

@app.get("/api/dataset/molecules")
def list_molecules(
    label: Optional[int] = Query(None, description="Filter by label 0 or 1"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=188)
):
    ds = get_dataset()
    all_items = []
    
    for i in range(len(ds)):
        data = ds[i]
        lbl = int(data.y.item())
        if label is not None and lbl != label:
            continue
        all_items.append({
            "id": i,
            "name": f"MUTAG #{i+1}",
            "num_nodes": data.x.size(0),
            "num_edges": data.edge_index.size(1) // 2,
            "label": lbl,
            "label_text": "Mutagenic" if lbl == 1 else "Non-Mutagenic"
        })
        
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated = all_items[start_idx:end_idx]
    
    return {
        "total": len(all_items),
        "page": page,
        "limit": limit,
        "total_pages": (len(all_items) + limit - 1) // limit,
        "molecules": paginated
    }

@app.get("/api/dataset/molecule/{molecule_id}")
def get_molecule(molecule_id: int):
    ds = get_dataset()
    if molecule_id < 0 or molecule_id >= len(ds):
        raise HTTPException(status_code=404, detail="Molecule ID not found in MUTAG dataset.")
        
    data = ds[molecule_id]
    graph_dict = mutag_to_json_graph(data, idx=molecule_id)
    
    # Run predictions across preloaded models
    predictions = {}
    for name in ["GIN", "GCN", "GRAPHSAGE", "GAT"]:
        try:
            cached = load_cached_model(name)
            model = cached["model"]
            expl = compute_node_importance(model, data, device="cpu")
            predictions[name] = {
                "predicted_class": expl["target_class"],
                "predicted_label": expl["target_label"],
                "p_non_mutagenic": round(expl["probabilities"][0], 4),
                "p_mutagenic": round(expl["probabilities"][1], 4),
                "confidence": round(max(expl["probabilities"]) * 100, 1),
                "node_importance": expl["importance_scores"]
            }
        except Exception as e:
            print(f"Error predicting with {name}: {e}")
            
    graph_dict["predictions"] = predictions
    return graph_dict

@app.get("/api/benchmarks")
def get_benchmarks():
    benchmark_file = os.path.join(WEIGHTS_DIR, "benchmark_results.json")
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    else:
        # Generate them
        from backend.train_eval import train_and_benchmark_all
        return train_and_benchmark_all()

@app.get("/api/presets")
def get_preset_molecules():
    return PRESET_MOLECULES

class PredictionRequest(BaseModel):
    smiles: Optional[str] = None
    mutag_id: Optional[int] = None
    model_name: str = "GIN"

@app.post("/api/predict")
def predict_molecule(req: PredictionRequest):
    cached = load_cached_model(req.model_name)
    model = cached["model"]
    
    if req.smiles:
        try:
            pyg_data, graph_dict = smiles_to_graph_and_pyg(req.smiles.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse SMILES molecule: {str(e)}")
    elif req.mutag_id is not None:
        ds = get_dataset()
        if req.mutag_id < 0 or req.mutag_id >= len(ds):
            raise HTTPException(status_code=404, detail="Invalid MUTAG ID")
        pyg_data = ds[req.mutag_id]
        graph_dict = mutag_to_json_graph(pyg_data, idx=req.mutag_id)
    else:
        raise HTTPException(status_code=400, detail="Must provide either smiles string or mutag_id")
        
    expl = compute_node_importance(model, pyg_data, device="cpu")
    
    # Attach importance to nodes for frontend heatmap
    importance_scores = expl["importance_scores"]
    for i, node in enumerate(graph_dict["nodes"]):
        if i < len(importance_scores):
            node["importance"] = importance_scores[i]
        else:
            node["importance"] = 0.0
            
    return {
        "graph": graph_dict,
        "model_used": req.model_name.upper(),
        "predicted_class": expl["target_class"],
        "predicted_label": expl["target_label"],
        "p_non_mutagenic": round(expl["probabilities"][0], 4),
        "p_mutagenic": round(expl["probabilities"][1], 4),
        "confidence_pct": round(max(expl["probabilities"]) * 100, 2),
        "is_mutagenic": expl["target_class"] == 1,
        "node_importance": importance_scores
    }

class TrainRequest(BaseModel):
    model_name: str = "GIN"
    hidden_dim: int = Field(64, ge=16, le=256)
    num_layers: int = Field(2, ge=1, le=5)
    pooling: str = Field("sum", pattern="^(mean|sum|max)$")
    learning_rate: float = Field(0.005, ge=0.0001, le=0.1)
    epochs: int = Field(40, ge=5, le=100)
    dropout: float = Field(0.2, ge=0.0, le=0.8)

@app.post("/api/train")
def train_model_endpoint(req: TrainRequest):
    try:
        res = train_single_model(
            model_name=req.model_name.upper(),
            epochs=req.epochs,
            lr=req.learning_rate,
            hidden_dim=req.hidden_dim,
            num_layers=req.num_layers,
            pooling=req.pooling,
            dropout=req.dropout,
            save_checkpoint=False  # Do not overwrite canonical benchmarks
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/research-report")
def get_research_report():
    return {
        "title": "Predicting Molecular Mutagenicity via Graph Neural Networks on MUTAG Benchmark",
        "domain_background": (
            "Chemical compounds are naturally modelled as attributed graphs where nodes represent individual atoms "
            "and edges denote chemical bonds. Biological activity, notably mutagenicity (the biochemical propensity to "
            "induce genetic mutations that frequently lead to carcinogenesis), is heavily determined by spatial "
            "connectivity and specific toxicophoric functional groups (e.g., nitroaromatic rings). Traditional cheminformatics "
            "often relied on handcrafted QSAR descriptors, whereas Graph Neural Networks (GNNs) learn end-to-end "
            "latent molecular representations by propagating relational chemical state across neighborhood topology."
        ),
        "hypotheses": [
            {
                "id": "H1",
                "title": "Structural Topology",
                "statement": "Topological arrangements of atomic bonds contain discriminative inductive signals sufficient to distinguish mutagenic from non-mutagenic compounds."
            },
            {
                "id": "H2",
                "title": "Message Passing Expressiveness",
                "statement": "Neighborhood message passing layers effectively propagate multi-hop chemical influence, allowing local atomic states to aggregate into coherent toxicophoric representations."
            },
            {
                "id": "H3",
                "title": "Architectural Invariance",
                "statement": "Different aggregation schemes (GCN isotropic convolution, GraphSAGE neighborhood sampling, GIN multi-set injectivity, and GAT anisotropic attention) exhibit varying inductive biases, with multiset-injective GIN showing superior performance on molecular graph classification."
            }
        ],
        "pipeline_steps": [
            {"step": 1, "title": "Molecular Graph Input", "desc": "Compounds parsed into attributed graph G = (V, E) with node features X (7 atom types) and edge adjacency E."},
            {"step": 2, "title": "Atom & Bond Embedding", "desc": "One-hot representation of atomic species [C, N, O, F, I, Cl, Br] and edge connectivity [aromatic, single, double, triple]."},
            {"step": 3, "title": "Message Passing (GNN)", "desc": "Multi-layer message aggregation: h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u in N(v)}))."},
            {"step": 4, "title": "Global Graph Pooling", "desc": "Permutation-invariant readout function z_G = POOL({h_v^(L) : v in V}) aggregating atomic embeddings into a fixed-dimension molecular embedding."},
            {"step": 5, "title": "Classification Head", "desc": "Multi-layer perceptron (MLP) mapping z_G to binary mutagenicity probability distribution via Softmax."}
        ],
        "limitations": [
            "Benchmark Scale: MUTAG consists of 188 molecular graphs, creating susceptibility to variance on small sample test splits.",
            "3D Conformation: 2D topological graphs lack 3D stereochemical and dihedral angle coordinates which can impact biological binding.",
            "Biological Association vs Causation: Learned representations identify strong statistical toxicophore correlations, but laboratory in-vitro Ames validation remains the gold standard."
        ]
    }

# Mount frontend build if available
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")

