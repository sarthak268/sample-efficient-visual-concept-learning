import torch

import numpy as np
from PIL import Image

from graph.graph import Graph, saveObject
from gsnn import gsnn
from detector.fasterrcnn_coco import detector
from utils.data_utils import makeLabelDict
from embeddings.glove_embeddings import get_embedding_matrix, use_glove_embeddings

def makeLabelDict(path):
    file = open(path, 'r')
    lines = file.readlines()
    
    nodelist = []
    for line in lines:  
        data = line.split(':')
        index = int(data[0])
        name = str(data[1])[:-1]
        nodelist.append(name)
    return nodelist

def detectClasses(opt, img_tensor, detection_net):
    boxes, pred_cls = detection_net.detect(img_tensor)
    return pred_cls

def checkEdgePresence(edge, edge_list):
    present = False
    idx = None
    for idx, ed in enumerate(edge_list):
        if ed[0] == edge[0]:
            if ed[1] == edge[1]:
                if ed[2] == edge[2]:
                    present = True
                    break
    return present, idx                

def findNode(node_idx, graph):
    for node in graph.nodes:
        if node.index == node_idx:
            return node    

def novelClassManualNodeAddition(relevant_node_names, graph):
    node_name_list = makeLabelDict('./nodename2index.txt')
    relevant_node_idx = [node_name_list.index(node) for node in relevant_node_names]
    edge = []

    for i in range(len(relevant_node_names)): 
        outgoing_edgetype = 0
        predicted_edgetype_outgoing = 1.
        new_edge = [findNode(relevant_node_idx[i], graph), outgoing_edgetype, 'in', predicted_edgetype_outgoing]
        edge.append(new_edge)

    return edge
  
def novelClassEdgeAddition(opt, novel_node_rep, novel_node_idx, novel_node_images, graph, gsnn_net, classifier, encoder_net):
    detection_net = detector(threshold = .3, model_path = None)
    node_name_list = makeLabelDict('nodename2index.txt')

    embeddings_index = use_glove_embeddings()
    embedding_matrix_vocab = get_embedding_matrix(node_name_list, embeddings_index)

    novel_node_rep = torch.from_numpy(novel_node_rep).unsqueeze(0).to(opt.device).float()

    edges = []

    gsnn_net = gsnn_net.to(opt.device)
    encoder_net = encoder_net.to(opt.device)

    for novel_node_image in novel_node_images:
        img_path = '/data/sarthak/VisualGenome/' + novel_node_image    
        image = Image.open(img_path)
        
        image_edge = image.resize((256, 256))
        if opt.load_net_type == 'VGG':
            image = image.resize((256, 256))
        elif opt.load_net_type == 'ViT':
            image = image.resize((384, 384))
        
        image = np.asarray(image).astype('float64')
        image_torch = torch.from_numpy(image).permute(2, 0, 1).float()
        image_torch = image_torch.to(opt.device).unsqueeze(0).to(opt.device)

        image_edge = np.asarray(image_edge).astype('float64')
        image_torch_edge = torch.from_numpy(image_edge).permute(2, 0, 1).float()
        image_torch_edge = image_torch_edge.to(opt.device).unsqueeze(0).to(opt.device)

        detected_classes = detectClasses(opt, image_torch, detection_net)
        # TODO: take detection confidence from faster rcnn into account
        cls_idx = [node_name_list.index(cls) for cls in detected_classes]
        initial_conf = torch.zeros((opt.vocab_size)).to(opt.device)
        initial_conf[cls_idx] = 1.
        
        annotations_plus = torch.zeros((opt.detector_size, 1)).to(opt.device)
            
        image_embedding = None
        if opt.image_cond_edge_pred:
            image_embedding = encoder_net(image_torch).squeeze(0)
        output, importance_outputs, reverse_lookup, active_idx, expanded_idx = \
                gsnn_net(graph, initial_conf, annotations_plus, evaluation=False, image_embedding=image_embedding)
        expanded_idx = list(set(cls_idx) | set(expanded_idx))

        for relevant_node in expanded_idx:
            
            if relevant_node < opt.vocab_size and relevant_node != node_name_list[novel_node_idx]:
            # NOTE: due to limited data we make sure we do not connect two novel nodes

                expanded_node_rep = torch.from_numpy(embedding_matrix_vocab[relevant_node]).unsqueeze(0).to(opt.device).float()
                
                if opt.image_cond_edge_pred:
                    predicted_edgetype_incoming = classifier(image_torch_edge, expanded_node_rep, novel_node_rep).squeeze(0)
                else:
                    predicted_edgetype_incoming = classifier(expanded_node_rep, novel_node_rep).squeeze(0)

                add_edge = False
                if predicted_edgetype_incoming[0] > opt.edge_confidence:
                    add_edge = True

                if add_edge:
                    incoming_edgetype = 0
                    conf_weight = 1.
                    if relevant_node not in cls_idx:
                        conf_weight *= opt.non_detected_expanded_concept_conf_weight
                    new_edge = [findNode(relevant_node, graph), incoming_edgetype, 'in', predicted_edgetype_incoming[0] * conf_weight]
                    pres, idx = checkEdgePresence(new_edge, edges)
                    if pres:
                        edges[idx][-1] += 1
                    else:
                        edges.append(new_edge)
                
                # Note: There is no outgoing edge when we are trying to add affordance and attribute
                # All affordances and attributes are assumed to be leaf nodes in KG
                if opt.image_cond_edge_pred:
                    predicted_edgetype_outgoing = classifier(image_torch_edge, novel_node_rep, expanded_node_rep).squeeze(0).float()
                else:
                    predicted_edgetype_outgoing = classifier(novel_node_rep, expanded_node_rep).squeeze(0)

                add_edge = False
                if predicted_edgetype_outgoing[0] > opt.edge_confidence:
                    add_edge = True

                if add_edge:
                    outgoing_edgetype = 0
                    conf_weight = 1.
                    if relevant_node not in cls_idx:
                        conf_weight *= opt.non_detected_expanded_concept_conf_weight
                    new_edge = [findNode(relevant_node, graph), outgoing_edgetype, 'out', predicted_edgetype_outgoing[0] * conf_weight]
                    pres, idx = checkEdgePresence(new_edge, edges)
                    if pres:
                        edges[idx][-1] += 1
                    else:
                        edges.append(new_edge)

    final_edges = []
    for e in edges:
        if e[-1].item() > opt.freq_threshold:
            final_edges.append(e)

    if opt.max_edges_allowed > 0:    
        if len(final_edges) < opt.max_edges_allowed:
            return final_edges
        else:
            confidences = []
            for i in range(len(final_edges)):
                confidences.append(final_edges[i][-1].item())

            final_edges_max = []
            for j in range(opt.max_edges_allowed):
                max_idx = np.argmax(confidences)
                confidences[max_idx] = 0.
                final_edges_max.append(final_edges[max_idx])
            return final_edges_max
    else:
        return final_edges

        