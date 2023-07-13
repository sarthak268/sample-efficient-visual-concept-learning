import torch

# Create all adjacency matrices for graph and concatenate them.
# The ordering of the adjacency matrices must be consistent with the ordering
# of the propagation networks.
def createAdjacencyMatrixCat(edges, n_nodes, n_edge_types, device):
    a = torch.zeros((n_nodes, n_nodes*n_edge_types*2)).to(device)
    for e in edges:
        src_idx, e_type, tgt_idx = e
        if (torch.is_tensor(src_idx)):
            src_idx = src_idx.item() - 1
            tgt_idx = tgt_idx.item() - 1
        a[tgt_idx, (e_type-1)*n_nodes+src_idx] = a[tgt_idx, (e_type-1)*n_nodes+src_idx] + 1
        a[src_idx, (e_type-1+n_edge_types)*n_nodes+tgt_idx] = a[src_idx, (e_type-1+n_edge_types)*n_nodes+tgt_idx] + 1
    return a
