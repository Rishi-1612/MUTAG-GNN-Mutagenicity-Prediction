import torch
import torch.nn.functional as F
import numpy as np

def compute_node_importance(model, data, target_class=None, device="cpu"):
    """
    Computes node-level saliency attribution using gradient of the target class
    logit with respect to input node feature representation.
    Highlights atoms that are most influential in predicting mutagenicity.
    """
    model.eval()
    x = data.x.clone().detach().to(device).float()
    x.requires_grad = True
    edge_index = data.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    
    out = model(x, edge_index, batch)
    probs = F.softmax(out, dim=1)
    
    if target_class is None:
        target_class = torch.argmax(probs, dim=1).item()
        
    score = out[0, target_class]
    score.backward()
    
    # Saliency is the L2 norm of the gradient with respect to node features
    with torch.no_grad():
        grad = x.grad
        if grad is not None:
            # Saliency per node
            saliency = torch.norm(grad, dim=1).cpu().numpy()
            # Normalize between 0 and 1
            max_val = np.max(saliency)
            min_val = np.min(saliency)
            if max_val > min_val:
                norm_saliency = (saliency - min_val) / (max_val - min_val)
            else:
                norm_saliency = np.ones_like(saliency) * 0.5
        else:
            norm_saliency = np.zeros(x.size(0))
            
    return {
        "target_class": int(target_class),
        "target_label": "Mutagenic" if target_class == 1 else "Non-Mutagenic",
        "probabilities": probs[0].detach().cpu().numpy().tolist(),
        "importance_scores": [round(float(s), 4) for s in norm_saliency]
    }
