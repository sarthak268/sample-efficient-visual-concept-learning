import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import time
import os
import copy
from itertools import cycle
import pickle
import numpy as np

from dataloader import data_graph
from args.args_continual import opt
from networks.networks_graph import VGG_Net, Classifier, classifierWeightsInit
from graph.graph import Graph, saveObject
from gsnn import gsnn
from utils.plot_utils import saveGraphNodes, saveImage
from utils.data_utils import makeLabelDict

def main():

    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(opt.seed)

    criterion = nn.BCELoss()
    importance_criterion = nn.MSELoss()

    criterion = criterion.to(device)
    importance_criterion = importance_criterion.to(device)

    graph = Graph(detector_size=opt.detector_size, vocab_size=opt.vocab_size)
    graph = pickle.load(open(opt.graph_pkl, 'rb'))

    nodes_removal_idx = []
    nodes_removal_idx_detect = []
    if opt.node_addition:
        novel_concept = opt.test_concepts
        graph.removeConnectionsNovelNode(novel_concept)
        print ('Removed {} nodes from graph'.format(len(novel_concept)))

        node_name_list = makeLabelDict('nodename2index.txt')
        nodes_removal_idx = [node_name_list.index(concept) for concept in novel_concept]
        
        detector_reverse_lookup = list(graph.detector_reverse_lookup)
        nodes_removal_idx_detect = [(detector_reverse_lookup.index(concept_idx) if concept_idx in detector_reverse_lookup \
                                    else -1) for concept_idx in nodes_removal_idx]
        nodes_removal_idx_detect =[val for val in nodes_removal_idx_detect if val > 0]

    if opt.dataset == 'vg':
        dataset = data_graph(opt=opt, data_dir='./filtered_data_train/', img_dir=opt.dataset_path, 
                            get_detection=True, take_subset=opt.take_subset, subset_ratio=opt.subset_ratio,
                            test_train_split=True, is_train=True, nodes_to_remove_vocab=nodes_removal_idx, 
                            nodes_to_remove_detections=nodes_removal_idx_detect)
        dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, drop_last=True)
    else:
        raise NotImplementedError

    if not os.path.exists('saved_models/' + opt.exp_name):
        os.makedirs('saved_models/' + opt.exp_name)

    graph.getNode2NodetypeMapping()
    graph.cleanGraph()

    if opt.load_net_type == 'VGG':
        encoder_net = VGG_Net()

        for param in encoder_net.parameters():
            param.requires_grad = False
        
        for param in encoder_net.vgg_network_classifier.parameters():
            param.requires_grad = True

    elif opt.load_net_type == 'ViT':
        from pytorch_pretrained_vit import ViT
        encoder_net = ViT('B_16_imagenet1k', pretrained=True)

        for param in encoder_net.parameters():
            param.requires_grad = False

        for param in encoder_net.fc.parameters():
            param.requires_grad = True

    else:
        raise Exception('Invalid load_net_type argument')

    # Loading GSNN params
    context_net_options = {}
    importance_net_options = {}
    context_net_options['architecture'] = opt.context_architecture
    context_net_options['transfer_function'] = opt.context_transfer_function
    context_net_options['use_node_input'] = opt.context_use_node_input
    context_net_options['use_annotation_input'] = opt.context_use_ann
    importance_net_options['architecture'] = opt.importance_architecture
    importance_net_options['transfer_function'] = opt.importance_transfer_function
    importance_net_options['use_node_input'] = opt.importance_use_node_input
    importance_net_options['use_annotation_input'] = opt.importance_use_ann
    importance_net_options['expand_type'] = 'value'
    
    gsnn_annotation_dim = 2  # Hack - annotation after conf is just 0's

    # Create graph net
    gsnn_net = gsnn.GSNN(opt, opt.state_dim, gsnn_annotation_dim, 
            graph.n_edge_types, opt.num_steps, opt.min_num_init, opt.context_dim, 
            opt.num_expand, opt.init_conf, graph.n_total_nodes, opt.node_bias_size, 
            opt.num_inter_steps, context_net_options, importance_net_options)

    net = Classifier(opt, len(nodes_removal_idx))
    net.apply(classifierWeightsInit)

    # optimizers
    if opt.optim == 'sgd':
        net_optim = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)
        encoder_optim = optim.SGD(encoder_net.parameters(), lr=opt.lr * opt.encoder_lr, momentum=opt.momentum)
    elif opt.optim == 'adam':
        net_optim = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        encoder_optim = optim.Adam(encoder_net.parameters(), lr=opt.lr * opt.encoder_lr, weight_decay=opt.weight_decay)

    if opt.optim_gsnn == 'sgd':
        gsnn_optim = optim.SGD(gsnn_net.parameters(), lr=opt.lr * opt.gsnn_lr, momentum=opt.momentum)
    elif opt.optim_gsnn == 'adam':
        gsnn_optim = optim.Adam(gsnn_net.parameters(), lr=opt.lr * opt.gsnn_lr, weight_decay=opt.weight_decay)

    net_scheduler = torch.optim.lr_scheduler.StepLR(net_optim, step_size=10, gamma=0.1)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optim, step_size=10, gamma=0.1)
    gsnn_scheduler = torch.optim.lr_scheduler.StepLR(gsnn_optim, step_size=10, gamma=0.1)

    net = net.to(device)
    encoder_net = encoder_net.to(device)
    gsnn_net = gsnn_net.to(device)

    if opt.start_epoch > 0 or opt.load_nets:
        net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/net.pth'))
        if opt.load_net_type == 'VGG':
            encoder_net.vgg_network_classifier.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/encoder_net_classifier.pth'))
        elif opt.load_net_type == 'ViT':
            encoder_net.fc.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/encoder_net_classifier.pth'))
        gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/gsnn_net.pth'))

    encoder_net.train()
    gsnn_net.train()

    # Initialize summary writer
    writer = SummaryWriter('runs/' + opt.exp_name)

    print ('Starting Training ...')

    if torch.cuda.is_available() and opt.device[:4] != 'cuda':
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    if not os.path.exists(opt.logs_dir):
        os.makedirs(opt.logs_dir)

    if not os.path.exists(opt.logs_dir + '/' + opt.exp_name):
        os.makedirs(opt.logs_dir + '/' + opt.exp_name)

    for epoch in range(opt.start_epoch, opt.num_epochs):
        
        for iteration, batch in enumerate(dataloader):

            net_optim.zero_grad()
            encoder_optim.zero_grad()
            gsnn_optim.zero_grad()

            multiclass_loss = 0.
            importance_loss = 0.
            
            if opt.dataset == 'vg':
                imdata, detectConf, lab = batch
                
                imdata = imdata.to(device)
                detectConf = detectConf.to(device)
                lab = lab.to(device)

                detect_conf = detectConf.squeeze(-1)

                image_data = copy.deepcopy(imdata)
                detect_annotation = copy.deepcopy(detect_conf)
                
                label = copy.deepcopy(lab) 
                label_cpu = lab.cpu() 

            elif opt.dataset == 'coco':
                image, labels_one_hot, detects_one_hot = batch

                image_data = image.to(device)
                lab = labels_one_hot.to(device)
                detect_conf = detects_one_hot.to(device)

                detect_annotation = copy.deepcopy(detect_conf)
                label = copy.deepcopy(lab) 
                label_cpu = lab.cpu() 

            # Classification net + encoder
            ############################################################################################

            # Forward through encoder net 
            encoder_out = encoder_net(image_data)

            graph_size = opt.context_dim * opt.vocab_size
            graph_data = torch.zeros((opt.batchsize, graph_size)).to(device)
            
            importance_outputs_batch = []
            reverse_lookup_batch = []
            active_idx_batch = []
            expanded_idx_batch = []

            # For each batch, do forward pass
            for i in range(opt.batchsize):

                batchAnnotation = detect_conf[i]
                initial_conf = batchAnnotation
                
                # Hack -- this basically ensures what scalar value we add to the second dim of annotation
                # All annotations are of the form of (num, 0)
                annotations_plus = torch.zeros((opt.detector_size, 1)).to(device)
                    
                # Forward through GSNN network
                image_embedding = None
                if opt.image_conditioned_propnet or opt.image_conditioned_propnet1:
                    image_embedding = encoder_out[i]
                output, importance_outputs, reverse_lookup, active_idx, expanded_idx = \
                        gsnn_net(graph, initial_conf, annotations_plus, evaluation=False, image_embedding=image_embedding)
                importance_outputs_batch.append(importance_outputs)
                reverse_lookup_batch.append(reverse_lookup)
                active_idx_batch.append(active_idx)
                expanded_idx_batch.append(expanded_idx[:-1])

                # Reorder output of graph
                # for j in range (len(active_idx)):
                for j in range (len(expanded_idx)):
                    # Get vocab index
                    # full_graph_idx = active_idx[j]
                    full_graph_idx = expanded_idx[j]
                    output_idx = graph.nodes[full_graph_idx].vocab_index

                    # If it's a vg vocab node, save its output
                    if output_idx != -1:
                        # Set correct part of graph_data
                        graph_data[i, (output_idx)*opt.context_dim: (output_idx + 1)*opt.context_dim] = output.squeeze(0)[j]
        
            # Forward pass
            output = net(encoder_out, detect_annotation, graph_data)

            # Get error
            multiclass_loss = opt.multiclass_loss_weight * criterion(output, label)
            
            # Importance net
            ##############################################################################################

            importance_gts = []

            for i in range(opt.batchsize):
                importance_outputs = importance_outputs_batch[i]
                active_idx = active_idx_batch[i]
                target_nodes = []
                for label_ind in range(label_cpu[i].shape[0]): 
                    if label_cpu[i][label_ind] > 0.5:
                        target_nodes.append(label_ind)

                if opt.node_addition:
                    nodes_removal_idx_sorted = sorted(nodes_removal_idx)
                    for idx in range(len(target_nodes)):
                        error_idx = 0
                        added = False
                        for removal_idx in range(len(nodes_removal_idx_sorted)):
                            if target_nodes[idx] < nodes_removal_idx_sorted[removal_idx]:
                                target_nodes[idx] += error_idx
                                added = True
                                break
                            else:
                                error_idx += 1
                        if not added:
                            target_nodes[idx] += error_idx

                importance_gt_orig_idx = graph.getDiscountedValues(target_nodes, opt.gamma, opt.num_steps)
                importance_gt = []
                for step in range(opt.num_steps - 1):
                    gt = torch.zeros((importance_outputs[step].shape)).to(device)
                    for ind in range(importance_outputs[step].shape[0]):
                        orig_idx = active_idx[ind]
                        gt[ind][0] = importance_gt_orig_idx[orig_idx]
                    importance_gt.append(gt)
                
                importance_gts.append(importance_gt)

            # Calculate importance losses
            importance_loss = 0.
            for i in range(opt.batchsize):
                importance_outputs = importance_outputs_batch[i]
                importance_gt = importance_gts[i]
                
                for step in range(opt.num_steps - 1):
                    il_step = importance_criterion(importance_outputs[step], importance_gt[step])
                    importance_loss += il_step
            importance_loss *= opt.importance_loss_weight
                    
            # Add up losses
            loss = multiclass_loss + importance_loss

            loss.backward()
            encoder_optim.step()
            gsnn_optim.step()
            net_optim.step()

            if iteration % opt.print_after == 0:
                print ('Epoch: [{} / {}], Iteration: [{} / {}], Multiclass Loss: {}, Importance Loss: {}, Total Loss: {}'.format(
                    epoch, opt.num_epochs, iteration, len(dataset) // opt.batchsize, multiclass_loss.item(), importance_loss.item(), loss.item()
                ))

            if iteration % opt.plot_after == 0:
                # Write to tensorboard
                writer.add_scalar('Multiclass Loss', multiclass_loss.item(),
                                epoch * (int(len(dataset) / opt.batchsize) + 1) + iteration)
                writer.add_scalar('Importance Loss', importance_loss.item(),
                                epoch * (int(len(dataset) / opt.batchsize) + 1) + iteration)
                writer.add_scalar('Total Loss', loss.item(),
                                epoch * (int(len(dataset) / opt.batchsize) + 1) + iteration)

        # Saving models
        if epoch % opt.save_after == 0:
            torch.save(net.state_dict(), './saved_models/' + opt.exp_name + '/net.pth')
            if opt.load_net_type == 'VGG':
                torch.save(encoder_net.vgg_network_classifier.state_dict(), './saved_models/' + opt.exp_name + '/encoder_net_classifier.pth')
            elif opt.load_net_type == 'ViT':
                torch.save(encoder_net.fc.state_dict(), './saved_models/' + opt.exp_name + '/encoder_net_classifier.pth')
            torch.save(gsnn_net.state_dict(), './saved_models/' + opt.exp_name + '/gsnn_net.pth')
            
            # Save graph for evaluation 
            Graph.__module__ = 'graph.graph'
            saveObject(graph, './saved_models/' + opt.exp_name + '/saved_graph.pkl')

        net_scheduler.step()
        encoder_scheduler.step()
        gsnn_scheduler.step()

if (__name__ == '__main__'):
    main()
