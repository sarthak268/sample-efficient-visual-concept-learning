import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import os
import random
import numpy as np
import copy
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score
from PIL import Image

from networks.networks_graph import Classifier_Single_Class, Classifier
from dataloader_pretrain import data_pretrain, data_pretrain_evaluation
from args.args_continual import opt
from networks.networks_graph import VGG_Net, Classifier
from graph.graph import Graph, saveObject
from gsnn import gsnn
from utils.data_utils import makeLabelDict, readList

def evaluate(opt, encoder_net, gsnn_net, classifier, graph, novel_class_idx=[], curr_vocab_size=-1):
    device = opt.device

    encoder_net = encoder_net.to(device)
    gsnn_net = gsnn_net.to(device)
    classifier = classifier.to(device)

    node_name_list = makeLabelDict('nodename2index.txt')
    # MSOCO 16 test classes
    test_classes = ['bicycle', 'boat', 'stop sign', 'bird', 'backpack',
                    'frisbee', 'snowboard', 'surfboard', 'cup', 'fork', 'spoon', 'broccoli',
                    'chair', 'keyboard', 'microwave', 'vase']

    test_classes_idx = [node_name_list.index(concept) for concept in test_classes]
    detector_reverse_lookup = list(graph.detector_reverse_lookup)
    test_classes_idx_detect = [(detector_reverse_lookup.index(concept_idx) if concept_idx in detector_reverse_lookup \
                                else -1) for concept_idx in test_classes_idx]
    test_classes_idx_detect =[val for val in test_classes_idx_detect if val > 0]

    remaining_class_idx = [idx for idx in test_classes_idx if (idx not in novel_class_idx)]

    dataset = data_pretrain_evaluation(opt, './filtered_data/', '/data/sarthak/VisualGenome/', \
                            novel_class_idx=novel_class_idx, test_classes_idx_detect=test_classes_idx_detect, \
                            remaining_class_idx=remaining_class_idx, only_novel_class=opt.evaluate_finetune_only_novel)
    dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, drop_last=True)

    print ('Beginning evaluation ...')
    
    for iteration, batch in enumerate(dataloader):
                
        img, detectConf, label = batch

        img = img.to(device)
        detectConf = detectConf.to(device)
        lab = label.to(device)

        image_data = copy.deepcopy(img)
        detect_conf = detectConf.squeeze(-1)
        detect_annotation = copy.deepcopy(detect_conf)
        label = copy.deepcopy(lab) 

        # Forward through encoder net 
        encoder_out = encoder_net(image_data)

        # Through graph net
        graph_size = opt.context_dim * opt.vocab_size
        graph_data = torch.zeros((opt.batchsize, graph_size)).to(device)

        total_loss = 0
        total_average_precision = 0
        total_precision = 0
        total_recall = 0
        total_accuracy = 0
        total_samples = 0
        correct_topk = 0
        total_samples_topk = 0
        topk = opt.top_k_score

        criterion = nn.BCELoss()

        # For each batch, do forward pass
        for i in range(opt.batchsize):

            batch_annotation = detect_conf[i]
            initial_conf = batch_annotation
            
            # Hack -- this basically ensures what scalar value we add to the second dim of annotation
            # All annotations are of the form of (num, 0)
            annotations_plus = torch.zeros((opt.detector_size, 1)).to(device)
                
            # Forward through GSNN network
            output, importance_outputs, reverse_lookup, active_idx, expanded_idx = \
                        gsnn_net(graph, initial_conf, annotations_plus, evaluation=False, 
                                    pretrain_gsnn=False, curr_vocab_size=curr_vocab_size)

            # Reorder output of graph
            for j in range (len(active_idx)):
                # Get vocab index
                full_graph_idx = active_idx[j]
                output_idx = graph.nodes[full_graph_idx].vocab_index

                # If it's a vg vocab node, save its output
                if output_idx != -1:
                    # Set correct part of graph_data_cpu
                    graph_data[i, (output_idx)*opt.context_dim: (output_idx + 1)*opt.context_dim] = output.squeeze(0)[j]

        # Forward pass
        output = classifier(encoder_out, detect_annotation, graph_data)
        if opt.evaluate_finetune_only_novel:
            output = output[:, novel_class_idx]
        output_cpu = output.detach().cpu().numpy()

        # Get gt label
        gt_label = label.to(device)
        if opt.evaluate_finetune_only_novel:
            gt_label = gt_label[:, novel_class_idx]
        gt_label_cpu = gt_label.cpu().numpy()

        classification_loss = criterion(output, gt_label)
        total_loss += classification_loss.item()

        for i in range(opt.batchsize):
            y_pred = output_cpu[i]
            y_true = gt_label_cpu[i]

            if np.any(y_true > 0):
                # average_precision = average_precision_score(y_true, y_pred, average='macro')
                # y_pred_one_hot = np.where(y_pred > 0.5, 1, 0)
                # precision = precision_score(y_true, y_pred_one_hot, average='macro')
                # recall = recall_score(y_true, y_pred_one_hot, average='macro')
                accuracy = accuracy_score(y_true, y_pred_one_hot) 
                # total_average_precision += average_precision 
                # total_precision += precision
                # total_recall += recall
                total_accuracy += accuracy
                top_preds_idx = y_pred.argsort()[-topk:][::-1]
                for preds in top_preds_idx:
                    if y_true[preds] == 1:
                        correct_topk += 1        
                
                total_samples += 1 
                total_samples_topk += topk

        if (iteration % opt.print_after == 0):
            print ('Iteration: [{} / {}], Avg. Accuracy: {}, TopK: {}, Total Loss: {}'.format(
                iteration, len(dataset) // opt.batchsize, 
                round(total_accuracy / len(dataloader), 5),
                round(correct_topk / total_samples_topk, 5), 
                round(total_loss / total_samples, 5)
            ))

    print ('Iteration: [{} / {}], Avg. Accuracy: {}, TopK: {}, Total Loss: {}'.format(
                iteration, len(dataset) // opt.batchsize,  
                round(correct_topk / total_samples_topk, 5),
                round(total_loss / total_samples, 5)
            ))
