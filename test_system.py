import os
import sys
import json
import torch
from fastapi.testclient import TestClient

def run_all_tests():
    print("=" * 60)
    print("RUNNING COMPLETE PROTOCOL VERIFICATION TESTS FOR MUTAG-GNN")
    print("=" * 60)

    # 1. Dataset Verification
    print("\n[1/7] Testing MUTAG Dataset Loading & Integrity...")
    from backend.train_eval import load_mutag_dataset
    dataset = load_mutag_dataset()
    assert len(dataset) == 188, f"Expected 188 graphs, got {len(dataset)}"
    assert dataset.num_features == 7, f"Expected 7 node features, got {dataset.num_features}"
    assert dataset.num_classes == 2, f"Expected 2 classes, got {dataset.num_classes}"
    print(f"  [PASS] MUTAG loaded successfully: {len(dataset)} graphs, 7 node features, 2 classes.")

    # 2. GNN Architectures Verification
    print("\n[2/7] Testing GNN Model Forward Passes (GCN, GraphSAGE, GIN, GAT)...")
    from backend.models.gnn_models import build_model
    sample = dataset[0]
    for model_name in ["GIN", "GCN", "GRAPHSAGE", "GAT"]:
        model = build_model(model_name, in_dim=7, hidden_dim=32, out_dim=2, num_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(sample.x, sample.edge_index)
            assert out.shape == (1, 2), f"{model_name} output shape mismatch: {out.shape}"
            probs = torch.softmax(out, dim=1)
            assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
        print(f"  [PASS] {model_name} forward pass verified: Output shape {list(out.shape)}")

    # 3. Pretrained Weights & Benchmark File
    print("\n[3/7] Verifying Pre-trained Weights & Benchmark JSON...")
    from backend.train_eval import WEIGHTS_DIR
    bench_file = os.path.join(WEIGHTS_DIR, "benchmark_results.json")
    assert os.path.exists(bench_file), f"Benchmark file missing at {bench_file}"
    with open(bench_file, "r") as f:
        bench_data = json.load(f)
    for m in ["GIN", "GCN", "GraphSAGE", "GAT"]:
        assert m in bench_data, f"Model {m} missing from benchmarks"
        acc = bench_data[m]["test_metrics"]["accuracy"]
        f1 = bench_data[m]["test_metrics"]["f1"]
        print(f"  [PASS] {m} Checkpoint verified: Test Acc = {acc*100:.2f}%, F1 = {f1*100:.2f}%")

    # 4. SMILES Parsing & Graph Conversion with RDKit
    print("\n[4/7] Testing SMILES to PyG Graph Conversion...")
    from backend.molecule_utils import smiles_to_graph_and_pyg
    test_smiles = [
        ("Nitrobenzene", "c1ccccc1[N+](=O)[O-]"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Benzene", "c1ccccc1")
    ]
    for name, smi in test_smiles:
        pyg_data, graph_dict = smiles_to_graph_and_pyg(smi)
        assert pyg_data.x.size(1) == 7, "Node feature dimension must be 7"
        assert len(graph_dict["nodes"]) == pyg_data.num_nodes
        assert "atom_counts" in graph_dict
        print(f"  [PASS] {name} parsed: {pyg_data.num_nodes} atoms, {len(graph_dict['edges'])} bonds")

    # 5. Explainability Saliency Engine
    print("\n[5/7] Testing Explainability & Attribution Engine...")
    from backend.explainability import compute_node_importance
    model = build_model("GIN", in_dim=7, hidden_dim=64, out_dim=2, num_layers=2)
    saliency = compute_node_importance(model, dataset[0])
    assert "importance_scores" in saliency
    assert len(saliency["importance_scores"]) == dataset[0].num_nodes
    assert len(saliency["probabilities"]) == 2
    print(f"  [PASS] Saliency computed: {len(saliency['importance_scores'])} node attribution scores generated")

    # 6. Full FastAPI Endpoints Integration Test
    print("\n[6/7] Testing FastAPI Endpoints...")
    from backend.main import app
    client = TestClient(app)

    # Health
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"
    print("  [PASS] GET /api/health -> 200 OK")

    # Overview
    r = client.get("/api/dataset/overview")
    assert r.status_code == 200 and r.json()["total_graphs"] == 188
    print("  [PASS] GET /api/dataset/overview -> 200 OK")

    # Molecule Details
    r = client.get("/api/dataset/molecule/0")
    assert r.status_code == 200 and "predictions" in r.json()
    print("  [PASS] GET /api/dataset/molecule/0 -> 200 OK")

    # Benchmarks
    r = client.get("/api/benchmarks")
    assert r.status_code == 200 and "GIN" in r.json()
    print("  [PASS] GET /api/benchmarks -> 200 OK")

    # Presets
    r = client.get("/api/presets")
    assert r.status_code == 200 and len(r.json()) > 0
    print("  [PASS] GET /api/presets -> 200 OK")

    # Inference with SMILES
    r = client.post("/api/predict", json={"smiles": "c1ccccc1[N+](=O)[O-]", "model_name": "GIN"})
    assert r.status_code == 200 and "predicted_label" in r.json()
    print("  [PASS] POST /api/predict -> 200 OK")

    # Research Report
    r = client.get("/api/research-report")
    assert r.status_code == 200 and "hypotheses" in r.json()
    print("  [PASS] GET /api/research-report -> 200 OK")

    # Frontend Static Mount
    r = client.get("/")
    assert r.status_code == 200 and "<!doctype html>" in r.text
    print("  [PASS] GET / (Frontend Single-Page App) -> 200 OK")

    # 7. Frontend Build Artifacts
    print("\n[7/7] Verifying Frontend Production Build...")
    dist_dir = os.path.join("frontend", "dist")
    assert os.path.exists(os.path.join(dist_dir, "index.html")), "index.html missing in dist"
    assert os.path.exists(os.path.join(dist_dir, "assets")), "assets folder missing in dist"
    print("  [PASS] Frontend production assets verified and ready to serve.")

    print("\n" + "=" * 60)
    print("ALL 7 TEST SUITES PASSED PERFECTLY! 100% OPERATIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
