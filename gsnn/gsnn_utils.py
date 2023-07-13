import torch

def createAnnTensorFromTable(annotations, n_nodes, annotation_dim, device):
    annotation_tensor = torch.zeros((n_nodes, annotation_dim)).to(device)
    for i in range(len(annotations)):
        for j in range (annotation_dim):
            annotation_tensor[i, j] = annotations[i][j]
    return annotation_tensor

# Convert node indices to proper lookup table input
def getLookuptableRep(active_idx, device):
    return torch.Tensor(active_idx).to(device)
