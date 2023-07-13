import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import random
import numpy as np
import copy
from sklearn.metrics import accuracy_score, average_precision_score
from PIL import Image

from networks.networks_graph import Classifier_Single_Class, Classifier
from dataloader_pretrain import data_pretrain, data_pretrain_evaluation
from args.args_continual import opt
from networks.networks_graph import VGG_Net, Classifier, singleClassifierWeightsInit
from graph.graph import Graph, saveObject
from gsnn import gsnn
import pickle
from utils.node_addition import getNodeConnections
# from pretrain_edgetype_classifier_1 import EdgeTypeClassifier
from utils.data_utils import makeLabelDict, readList
from embeddings.clip_embeddings import find_relevant

def pretrainNetworks(pretrain_net, opt, novel_node_image, encoder_net, gsnn_net, classifier, graph, 
                        key_concepts=[], prev_images=[],
                        curr_vocab_size=-1):

    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists('saved_models/' + opt.exp_name):
        os.makedirs('saved_models/' + opt.exp_name)

    node_name_list = makeLabelDict('nodename2index.txt')
    test_classes = ['bicycle', 'boat', 'stop sign', 'bird', 'backpack',
                        'frisbee', 'snowboard', 'surfboard', 'cup', 'fork', 'spoon', 'broccoli',
                        'chair', 'keyboard', 'microwave', 'vase']

    test_classes_idx = [node_name_list.index(concept) for concept in test_classes]
    detector_reverse_lookup = list(graph.detector_reverse_lookup)
    test_classes_idx_detect = [(detector_reverse_lookup.index(concept_idx) if concept_idx in detector_reverse_lookup \
                                else -1) for concept_idx in test_classes_idx]
    test_classes_idx_detect =[val for val in test_classes_idx_detect if val > 0]

    novel_class_ratio = opt.novel_class_ratio
    dataset = data_pretrain(opt, './filtered_data/', '/data/sarthak/VisualGenome/', novel_node_image, 
                                novel_class_ratio=novel_class_ratio, 
                                novel_class_idx=test_classes_idx, novel_class_idx_detect=test_classes_idx_detect, 
                                prev_images=prev_images)
    dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, drop_last=True)
    
    test_dataset = data_pretrain(opt, './filtered_data/', '/data/sarthak/VisualGenome/', novel_node_image, 
                                    novel_class_ratio=novel_class_ratio, 
                                    novel_class_idx=test_classes_idx, novel_class_idx_detect=test_classes_idx_detect,
                                    prev_images=prev_images)
    test_dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, drop_last=True)

    if pretrain_net == 'classifier' or pretrain_net == 'classifier_only' or pretrain_net == 'modified_gsnn':
        final_classifier = Classifier_Single_Class(opt)
        final_classifier.apply(singleClassifierWeightsInit)
    elif pretrain_net == 'gsnn':
        final_classifier = classifier
    elif pretrain_net == 'both':
        final_classifier = Classifier_Single_Class(opt)
        final_classifier.apply(singleClassifierWeightsInit)
    else:
        raise Exception('Invalid pretrain_net')

    encoder_net = encoder_net.to(device)
    gsnn_net = gsnn_net.to(device)
    final_classifier = final_classifier.to(device)

    encoder_net.eval()
    gsnn_net.importance_out_net.eval()
    gsnn_net.context_out_net.eval()
    
    if pretrain_net == 'gsnn':
        optimizer_gsnn = optim.Adam(list(gsnn_net.baseggnns.parameters()),
                        lr=opt.finetune_lr_gsnn, weight_decay=opt.weight_decay)
        optimizer_bias = optim.Adam(list(gsnn_net.context_out_net.embedding_node_bias_novel.parameters()) + 
                        list(gsnn_net.importance_out_net.embedding_node_bias_novel.parameters()), \
                        lr=opt.finetune_lr_bias, weight_decay=opt.weight_decay)
    
    elif pretrain_net == 'classifier' or pretrain_net == 'classifier_only' or pretrain_net == 'modified_gsnn':
        optimizer_classifier = optim.Adam(final_classifier.parameters(), lr=opt.finetune_lr_classifier, weight_decay=opt.weight_decay)

    elif pretrain_net == 'both':
        optimizer_classifier = optim.Adam(final_classifier.parameters(), lr=opt.finetune_lr_classifier, weight_decay=opt.weight_decay)
        optimizer_gsnn = optim.Adam(list(gsnn_net.baseggnns.parameters()),
                        lr=opt.finetune_lr_gsnn, weight_decay=opt.weight_decay)

    criterion = nn.BCELoss()

    for epoch in range(opt.num_epochs_fine):

        if pretrain_net == 'gsnn':
            gsnn_net.baseggnns.train()
            gsnn_net.importance_out_net.embedding_node_bias_novel.train()
            gsnn_net.context_out_net.embedding_node_bias_novel.train()
            final_classifier.eval()

        elif pretrain_net == 'classifier' or pretrain_net == 'classifier_only' or pretrain_net == 'modified_gsnn':
            final_classifier.train()
            gsnn_net.eval()

        elif pretrain_net == 'both':
            gsnn_net.baseggnns.train()      
            final_classifier.train()
        
        for iteration, batch in enumerate(dataloader):

            print(iteration, '=======================')
                
            img, detect_conf, label, is_novel_class = batch

            img = img.to(device)
            detect_conf = detect_conf.to(device)
            lab = label.to(device)

            image_data = copy.deepcopy(img)
            detect_conf = detect_conf.squeeze(-1)
            detect_annotation = copy.deepcopy(detect_conf)
            label = copy.deepcopy(lab) 

            # Forward through encoder net 
            encoder_out = encoder_net(image_data)

            # Through graph net
            graph_size = opt.context_dim * opt.vocab_size
            graph_data = torch.zeros((opt.batchsize, graph_size)).to(device)
        
            # For each batch, do forward pass
            for i in range(opt.batchsize):

                batch_annotation = detect_conf[i]
                initial_conf = batch_annotation
                
                # Hack -- this basically ensures what scalar value we add to the second dim of annotation
                # All annotations are of the form of (num, 0)
                annotations_plus = torch.zeros((opt.detector_size, 1)).to(device)

                novel_class_sample = is_novel_class[i]
                    
                pretrain_gsnn = pretrain_net == 'gsnn' and novel_class_sample

                image_embedding = encoder_out[i]
                image_emb = image_embedding if opt.image_conditioned_propnet or opt.image_conditioned_propnet1 else None
                
                # Forward through GSNN network
                output, importance_outputs, reverse_lookup, active_idx, expanded_idx = \
                            gsnn_net(graph, initial_conf, annotations_plus, evaluation=False, 
                                        pretrain_gsnn=pretrain_gsnn, curr_vocab_size=curr_vocab_size,
                                        image_embedding=image_emb)
                
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
            output = final_classifier(encoder_out, detect_annotation, graph_data)

            # Get error and gradient
            gt_label = is_novel_class.float().unsqueeze(-1).to(device) \
                if (pretrain_net == 'classifier' or pretrain_net == 'both' or pretrain_net == 'classifier_only' or pretrain_net == 'modified_gsnn') \
                else label.to(device)

            classification_loss = criterion(output, gt_label)

            if pretrain_net == 'gsnn':
                optimizer_gsnn.zero_grad()
                optimizer_bias.zero_grad()

            elif pretrain_net == 'classifier' or pretrain_net == 'classifier_only':
                optimizer_classifier.zero_grad()

            elif pretrain_net == 'both':
                optimizer_gsnn.zero_grad()
                optimizer_classifier.zero_grad()
            
            classification_loss.backward()
            
            if pretrain_net == 'gsnn':
                optimizer_gsnn.step()
                optimizer_bias.step()

            elif pretrain_net == 'classifier' or pretrain_net == 'classifier_only':
                optimizer_classifier.step()

            elif pretrain_net == 'both':
                optimizer_gsnn.step()
                optimizer_classifier.step()

            if iteration % opt.print_after_fine == 0:
                print ('Epoch: [{} / {}], Iteration: [{} / {}], Total Loss: {}'.format(
                    epoch, opt.num_epochs_fine, iteration, len(dataset) // opt.batchsize, round(classification_loss.item(), 5)
                ))
        
        if opt.evaluate_finetune:

            if pretrain_net == 'gsnn':
                gsnn_net.baseggnns.eval()
                gsnn_net.importance_out_net.embedding_node_bias_novel.eval()
                gsnn_net.context_out_net.embedding_node_bias_novel.eval()

            elif pretrain_net == 'classifier' or pretrain_net == 'classifier_only':
                final_classifier.eval()

            elif pretrain_net == 'both':
                gsnn_net.baseggnns.eval()
                final_classifier.eval()

            total_score = 0
            total_samples = 0
            total_loss = 0

            for iteration, batch in enumerate(test_dataloader):
                    
                img, detectConf, detectClass, label, is_novel_class = batch

                img = img.to(device)
                detectConf = detectConf.to(device)
                detectClass = detectClass.to(device)

                image_data = copy.deepcopy(img)
                detect_conf = detectConf.squeeze(-1)
                detect_annotation = copy.deepcopy(detect_conf)

                # Forward through encoder net 
                encoder_out = encoder_net(image_data)

                # Through graph net
                graph_size = opt.context_dim * opt.vocab_size
                graph_data = torch.zeros((opt.batchsize, graph_size)).to(device)
            
                # For each batch, do forward pass
                for i in range(opt.batchsize):

                    batchAnnotation = detect_conf[i]
                    initial_conf = batchAnnotation
                    
                    # Hack -- this basically ensures what scalar value we add to the second dim of annotation
                    # All annotations are of the form of (num, 0)
                    annotations_plus = torch.zeros((opt.detector_size, 1)).to(device)

                    novel_class_sample = is_novel_class[i]
                        
                    # Forward through GSNN network
                    pretrain_gsnn = pretrain_net == 'gsnn' and novel_class_sample
                    output, importance_outputs, reverse_lookup, active_idx, expanded_idx = \
                                gsnn_net(graph, initial_conf, annotations_plus, evaluation=False, pretrain_gsnn=pretrain_gsnn)

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
                output = final_classifier(encoder_out, detect_annotation, graph_data)
                output_cpu = output.detach().cpu().numpy()

                # Get gt label
                gt_label = is_novel_class.float().unsqueeze(-1).to(device) \
                                if (pretrain_net == 'classifier' or pretrain_net == 'both' or pretrain_net == 'classifier_only') \
                                else label.to(device)
                gt_label_cpu = gt_label.cpu().numpy()

                classification_loss = criterion(output, gt_label)
                total_loss += classification_loss.item()

                for i in range(opt.batchsize):
                    y_pred = output_cpu[i]
                    y_true = gt_label_cpu[i]

                    if pretrain_net == 'gsnn':
                        if np.any(y_true > 0):
                            average_score = average_precision_score(y_true, y_pred, average='micro') 
                            total_score += average_score 
                            total_samples += 1 

                    elif pretrain_net == 'classifier' or pretrain_net == 'both' or pretrain_net == 'classifier_only':
                        y_true = y_true[0]
                        y_pred = float(y_pred[0] > 0.5)
                        if y_true == y_pred:
                            total_score += 1
                        total_samples += 1

            print ('Epoch: [{} / {}], Avg. Score: {}, Total Loss: {}'.format(
                epoch, opt.num_epochs_fine, round(total_score / total_samples, 5), round(total_loss / total_samples, 5)
            ))

        # Saving models
        if epoch % opt.save_after == 0:
            if pretrain_net == 'classifier' or pretrain_net == 'classifier_only' or pretrain_net == 'modified_gsnn':
                torch.save(final_classifier.state_dict(), './saved_models/' + opt.exp_name + '/novel_class_classifier.pth')

            elif pretrain_net == 'gsnn':
                torch.save(gsnn_net.state_dict(), './saved_models/' + opt.exp_name + '/novel_class_gsnn.pth')    
            
            elif pretrain_net == 'both':
                torch.save(final_classifier.state_dict(), './saved_models/' + opt.exp_name + '/novel_class_classifier.pth')
                torch.save(gsnn_net.state_dict(), './saved_models/' + opt.exp_name + '/novel_class_gsnn.pth')    

    if pretrain_net == 'gsnn':
        return gsnn
    
    elif pretrain_net == 'classifier' or pretrain_net == 'classifier_only':
        return final_classifier

    elif pretrain_net == 'both':
        return gsnn, final_classifier

def trainSingleConcept(pretrain_net, novel_img_list, gsnn_net, encoder_net, net, graph, 
                            prev_images, curr_vocab_size):

    if pretrain_net == 'seq':
        
        # pretrain step 1: GSNN
        print ('Training GSNN alone...')
        pretrained_network_1 = pretrainNetworks(pretrain_net='gsnn', opt=opt, 
                                novel_node_image=novel_img_list, encoder_net=encoder_net, 
                                gsnn_net=gsnn_net, classifier=net, graph=graph, 
                                prev_images=prev_images, curr_vocab_size=curr_vocab_size)

        gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_gsnn.pth'))
        gsnn_net.importance_out_net.updateNodeEmbedding()
        gsnn_net.context_out_net.updateNodeEmbedding()
        torch.save(gsnn_net.state_dict(), './saved_models/' + opt.exp_name + '/final_gsnn.pth')

        # pretrain step 2: both (GSNN + classifier)
        print ('Training both GSNN and classifier...')
        pretrained_network_2 = pretrainNetworks(pretrain_net='both', opt=opt, 
                                novel_node_image=novel_img_list, encoder_net=encoder_net, 
                                gsnn_net=gsnn_net, classifier=net, graph=graph, 
                                prev_images=prev_images, curr_vocab_size=curr_vocab_size)

        
        gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_gsnn.pth'))
        torch.save(gsnn_net.state_dict(), './saved_models/' + opt.exp_name + '/final_gsnn.pth')

        single_class_classifier = Classifier_Single_Class(opt)
        single_class_classifier.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_classifier.pth'))
        net.updateClassifierNovelClass(single_class_classifier=single_class_classifier)
        torch.save(net.state_dict(), './saved_models/' + opt.exp_name + '/final_classifier.pth')
        
    else:
        pretrained_network = pretrainNetworks(pretrain_net=pretrain_net, opt=opt, 
                                novel_node_image=novel_img_list, encoder_net=encoder_net, 
                                gsnn_net=gsnn_net, classifier=net, graph=graph, 
                                prev_images=prev_images, curr_vocab_size=curr_vocab_size)
        
        if pretrain_net == 'classifier' or pretrain_net == 'classifier_only' or pretrain_net == 'modified_gsnn':
            single_class_classifier = Classifier_Single_Class(opt)
            single_class_classifier.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_classifier.pth'))
            net.updateClassifierNovelClass(single_class_classifier=single_class_classifier)
            torch.save(net.state_dict(), './saved_models/' + opt.exp_name + '/final_classifier.pth')

        elif pretrain_net == 'gsnn':
            gsnn_net.importance_out_net.getNodeBiasNovel()
            gsnn_net.context_out_net.getNodeBiasNovel()
            gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_gsnn.pth'))
            gsnn_net.importance_out_net.updateNodeEmbedding()
            gsnn_net.context_out_net.updateNodeEmbedding()
            torch.save(gsnn_net.state_dict(), './saved_models/' + opt.exp_name + '/final_gsnn.pth')

        elif pretrain_net == 'both':
            single_class_classifier = Classifier_Single_Class(opt)
            single_class_classifier.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_classifier.pth'))
            net.updateClassifierNovelClass(single_class_classifier=single_class_classifier)
            torch.save(net.state_dict(), './saved_models/' + opt.exp_name + '/final_classifier.pth')
            
            gsnn_net.importance_out_net.getNodeBiasNovel()
            gsnn_net.context_out_net.getNodeBiasNovel()
            gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_gsnn.pth'))
            gsnn_net.importance_out_net.updateNodeEmbedding()
            gsnn_net.context_out_net.updateNodeEmbedding()
            torch.save(gsnn_net.state_dict(), './saved_models/' + opt.exp_name + '/final_gsnn.pth')

    return gsnn_net, encoder_net, net, graph
