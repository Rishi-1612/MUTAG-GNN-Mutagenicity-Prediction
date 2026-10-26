import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

# Atomic mapping for MUTAG 7 node features:
# Index: 0 -> C, 1 -> N, 2 -> O, 3 -> F, 4 -> I, 5 -> Cl, 6 -> Br
ATOM_SYMBOLS = ["C", "N", "O", "F", "I", "Cl", "Br"]
ATOM_NAMES = {
    "C": "Carbon",
    "N": "Nitrogen",
    "O": "Oxygen",
    "F": "Fluorine",
    "I": "Iodine",
    "Cl": "Chlorine",
    "Br": "Bromine"
}
ATOM_COLORS = {
    "C": "#64748B",  # Slate
    "N": "#3B82F6",  # Blue
    "O": "#EF4444",  # Red
    "F": "#10B981",  # Emerald
    "I": "#8B5CF6",  # Purple
    "Cl": "#06B6D4", # Cyan
    "Br": "#F59E0B"  # Amber
}

# Bond mapping for MUTAG 4 edge features:
# Index: 0 -> Aromatic, 1 -> Single, 2 -> Double, 3 -> Triple
BOND_TYPES = ["Aromatic", "Single", "Double", "Triple"]

PRESET_MOLECULES = [
    {
        "id": "nitrobenzene",
        "name": "Nitrobenzene",
        "smiles": "c1ccccc1[N+](=O)[O-]",
        "category": "Nitroaromatic",
        "description": "Prototype mutagenic compound with an active nitro group on a benzene ring."
    },
    {
        "id": "2-nitrofluorene",
        "name": "2-Nitrofluorene",
        "smiles": "O=[N+]([O-])c1ccc2c(c1)Cc3ccccc23",
        "category": "Polycyclic Aromatic Nitro",
        "description": "Potent direct-acting mutagen formed during combustion and diesel exhaust."
    },
    {
        "id": "4-nitrobiphenyl",
        "name": "4-Nitrobiphenyl",
        "smiles": "O=[N+]([O-])c1ccc(cc1)c2ccccc2",
        "category": "Bi-aromatic Nitro",
        "description": "Classic Ames test positive mutagen with strong DNA-binding capability."
    },
    {
        "id": "benzene",
        "name": "Benzene",
        "smiles": "c1ccccc1",
        "category": "Simple Aromatic",
        "description": "Simple non-substituted aromatic ring (non-mutagenic in classical bacterial assays)."
    },
    {
        "id": "aspirin",
        "name": "Aspirin (Acetylsalicylic Acid)",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "category": "Pharmaceutical",
        "description": "Widely used anti-inflammatory drug, negative for bacterial mutagenicity."
    },
    {
        "id": "aniline",
        "name": "Aniline",
        "smiles": "Nc1ccccc1",
        "category": "Aromatic Amine",
        "description": "Amino-substituted aromatic ring without nitro-group activation."
    },
    {
        "id": "chlorobenzene",
        "name": "Chlorobenzene",
        "smiles": "Clc1ccccc1",
        "category": "Halogenated Aromatic",
        "description": "Monochlorinated benzene ring used as industrial solvent."
    }
]

def mutag_to_json_graph(data: Data, idx: int = 0):
    num_nodes = data.x.size(0)
    
    # Build NetworkX graph for clean layout
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
        
    edges = data.edge_index.t().cpu().numpy().tolist()
    edge_types = {}
    if data.edge_attr is not None:
        attr_np = data.edge_attr.cpu().numpy()
        for i, (u, v) in enumerate(edges):
            if u < v:  # undirected canonical
                b_type_idx = int(np.argmax(attr_np[i]))
                edge_types[(u, v)] = BOND_TYPES[b_type_idx] if b_type_idx < len(BOND_TYPES) else "Single"
                G.add_edge(u, v)
    else:
        for u, v in edges:
            if u < v:
                G.add_edge(u, v)
                edge_types[(u, v)] = "Single"

    # Layout coordinates
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42 + idx, k=0.8, iterations=50)

    # Normalize coordinates to range 50 -> 450
    coords = np.array(list(pos.values()))
    if len(coords) > 0:
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        span = max_c - min_c
        span[span == 0] = 1.0
        normalized = (coords - min_c) / span
    else:
        normalized = np.zeros((num_nodes, 2))

    nodes_json = []
    for i in range(num_nodes):
        feat = data.x[i].cpu().numpy()
        atom_type_idx = int(np.argmax(feat))
        symbol = ATOM_SYMBOLS[atom_type_idx] if atom_type_idx < len(ATOM_SYMBOLS) else "C"
        x_coord = float(50 + normalized[i][0] * 380)
        y_coord = float(50 + normalized[i][1] * 380)
        
        nodes_json.append({
            "id": i,
            "label": f"{symbol}{i+1}",
            "symbol": symbol,
            "name": ATOM_NAMES.get(symbol, "Unknown"),
            "color": ATOM_COLORS.get(symbol, "#64748B"),
            "x": round(x_coord, 1),
            "y": round(y_coord, 1),
            "degree": int(G.degree(i)),
            "features": feat.tolist()
        })

    edges_json = []
    seen = set()
    for u, v in G.edges():
        pair = (min(u, v), max(u, v))
        if pair not in seen:
            seen.add(pair)
            b_type = edge_types.get(pair, "Single")
            edges_json.append({
                "source": pair[0],
                "target": pair[1],
                "type": b_type,
                "is_aromatic": b_type == "Aromatic",
                "is_double": b_type == "Double"
            })

    label_val = int(data.y.item()) if data.y is not None else 0
    return {
        "id": idx,
        "name": f"MUTAG #{idx+1}",
        "num_nodes": num_nodes,
        "num_edges": len(edges_json),
        "label": label_val,
        "label_text": "Mutagenic" if label_val == 1 else "Non-Mutagenic",
        "nodes": nodes_json,
        "edges": edges_json,
        "atom_counts": {s: sum(1 for n in nodes_json if n["symbol"] == s) for s in ATOM_SYMBOLS}
    }

def smiles_to_graph_and_pyg(smiles_str: str):
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_str}")
        
    mol = Chem.AddHs(mol)  # Add hydrogens or keep heavy? MUTAG has implicit H or explicit? MUTAG has 17 atoms for Benzene derivative so it includes heavy atoms + key H
    # Actually MUTAG graphs are heavy-atom graphs with nitro oxygens/nitrogens!
    # Let's remove explicit H to match MUTAG style (only heavy atoms: C, N, O, F, I, Cl, Br)
    mol = Chem.RemoveHs(mol)
    
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    
    num_nodes = mol.GetNumAtoms()
    node_features = []
    nodes_json = []
    
    # Coordinates
    coords = []
    for i in range(num_nodes):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y])
    coords = np.array(coords)
    if len(coords) > 0:
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        span = max_c - min_c
        span[span == 0] = 1.0
        normalized = (coords - min_c) / span
    else:
        normalized = np.zeros((num_nodes, 2))
        
    for i in range(num_nodes):
        atom = mol.GetAtomWithIdx(i)
        sym = atom.GetSymbol()
        feat = [0.0] * 7
        if sym in ATOM_SYMBOLS:
            feat[ATOM_SYMBOLS.index(sym)] = 1.0
        else:
            feat[0] = 1.0  # fallback to Carbon
            sym = "C"
            
        node_features.append(feat)
        x_coord = float(50 + normalized[i][0] * 380)
        y_coord = float(50 + (1.0 - normalized[i][1]) * 380)  # flip y for SVG
        
        nodes_json.append({
            "id": i,
            "label": f"{sym}{i+1}",
            "symbol": sym,
            "name": ATOM_NAMES.get(sym, sym),
            "color": ATOM_COLORS.get(sym, "#64748B"),
            "x": round(x_coord, 1),
            "y": round(y_coord, 1),
            "degree": atom.GetDegree(),
            "features": feat
        })
        
    edge_index_list = []
    edges_json = []
    edge_attr_list = []
    
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        
        # Bond type
        b_type = "Single"
        attr = [0.0, 1.0, 0.0, 0.0]  # [aromatic, single, double, triple]
        if bond.GetIsAromatic():
            b_type = "Aromatic"
            attr = [1.0, 0.0, 0.0, 0.0]
        elif bond.GetBondTypeAsDouble() == 2.0:
            b_type = "Double"
            attr = [0.0, 0.0, 1.0, 0.0]
        elif bond.GetBondTypeAsDouble() == 3.0:
            b_type = "Triple"
            attr = [0.0, 0.0, 0.0, 1.0]
            
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])
        edge_attr_list.append(attr)
        edge_attr_list.append(attr)
        
        edges_json.append({
            "source": u,
            "target": v,
            "type": b_type,
            "is_aromatic": b_type == "Aromatic",
            "is_double": b_type == "Double"
        })
        
    if len(edge_index_list) == 0:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr_list, dtype=torch.float)
        
    x_tensor = torch.tensor(node_features, dtype=torch.float)
    
    pyg_data = Data(
        x=x_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        num_nodes=num_nodes
    )
    
    graph_json = {
        "id": "custom",
        "name": smiles_str,
        "smiles": smiles_str,
        "num_nodes": num_nodes,
        "num_edges": len(edges_json),
        "nodes": nodes_json,
        "edges": edges_json,
        "atom_counts": {s: sum(1 for n in nodes_json if n["symbol"] == s) for s in ATOM_SYMBOLS}
    }
    
    return pyg_data, graph_json
