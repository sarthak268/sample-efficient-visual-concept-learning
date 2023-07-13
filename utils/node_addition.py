import torch
import numpy as np
from utils.data_utils import makeLabelDict, getNodeMapping
from embeddings.glove_embeddings import use_glove_embeddings, get_embedding_matrix, get_embedding_matrix_database

def sortUsingConfidence(arr, conf_arr):
    sorting_idx = np.argsort(conf_arr)
    arr_sorted = [arr[idx] for idx in sorting_idx]
    return arr_sorted

def getNodeConnections(opt, node_name, autoencoder, edgetype_classifier, graph, device, use_autoencoder=False):

    embeddings_index = use_glove_embeddings()
    index2node = makeLabelDict('nodename2index_corrected.txt') 
    embedding_matrix_vocab = get_embedding_matrix(index2node, embeddings_index)
    embedding_matrix, database_word_list = get_embedding_matrix_database(embeddings_index)
    
    edges_to_be_added = []
    confidence_list = []

    for node in graph.nodes:
        if node.index < opt.vocab_size:
            graph_node_glove = torch.from_numpy(embedding_matrix_vocab[node.index]).float().to(device)
        else:
            ind = database_word_list.index(getNodeMapping(node.name))
            graph_node_glove = torch.from_numpy(embedding_matrix[ind]).float().to(device)
        novel_node_index = database_word_list.index(getNodeMapping(node_name))
        novel_node_glove = torch.from_numpy(embedding_matrix[novel_node_index]).float().to(device)

        if use_autoencoder:
            auto_input = torch.cat((graph_node_glove.unsqueeze(0), novel_node_glove.unsqueeze(0)), dim=0)
            node_rep, node_recon = autoencoder(auto_input)
            node1_rep, node2_rep = node_rep[0], node_rep[1]
        else:
            node1_rep = graph_node_glove
            node2_rep = novel_node_glove

        edgetype_pred = edgetype_classifier(node1_rep, node2_rep).squeeze(0) 
        significant_edges = torch.gt(edgetype_pred, opt.edgetype_threshold).float()
        for i in range(significant_edges.shape[0]):
            if significant_edges[i] == 1: 
                edges_to_be_added.append([node, i, 'in', edgetype_pred[i].item()])
                confidence_list.append(edgetype_pred[i].item())

        edgetype_pred = edgetype_classifier(node2_rep, node1_rep).squeeze(0) 
        significant_edges = torch.gt(edgetype_pred, opt.edgetype_threshold).float()
        for i in range(significant_edges.shape[0]):
            if significant_edges[i] == 1: 
                edges_to_be_added.append([node, i, 'out', edgetype_pred[i].item()])
                confidence_list.append(edgetype_pred[i].item())

    sorted_edges_to_be_added = sortUsingConfidence(edges_to_be_added, confidence_list)[::-1]

    return sorted_edges_to_be_added


