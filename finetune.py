import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import glob
import os
import random
import numpy as np
import copy
import pickle
from sklearn.metrics import accuracy_score, average_precision_score
from PIL import Image

from networks.networks_graph import Classifier_Single_Class, Classifier
from dataloader_pretrain import data_pretrain, data_pretrain_evaluation
from args.args_continual import opt
from networks.networks_graph import VGG_Net, Classifier, singleClassifierWeightsInit
from graph.graph import Graph, saveObject
from gsnn import gsnn
from utils.node_addition import getNodeConnections
from utils.data_utils import makeLabelDict, readList
from embeddings.clip_embeddings import find_relevant
from novel_class_addition.evaluate_fine import evaluate
from novel_class_addition.pretrain_nets import pretrainNetworks, trainSingleConcept
from novel_class_addition.edge_addition import novelClassEdgeAddition, novelClassManualNodeAddition
from embeddings.glove_embeddings import get_embedding_matrix_database, use_glove_embeddings
from utils.data_utils import getNodeMapping

def main():
    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(opt.seed)

    graph = Graph(detector_size=opt.detector_size, vocab_size=opt.vocab_size)
    graph = pickle.load(open(opt.graph_pkl, 'rb'))
    graph.getNode2NodetypeMapping()
    graph.cleanGraph()

    if not opt.use_original_edges:
        if opt.use_multimodal_edge_pred:
            from transformer.relate import EdgeTransformer
            edgetype_classifier = EdgeTransformer(image_size = 256,
                                patch_size = 32,
                                num_classes = 1,
                                dim = opt.multimodal_attention_dim,
                                word_dim=opt.node_embedding_dim,
                                depth = opt.multimodal_attention_depth,
                                heads = opt.multimodal_attention_num_heads,
                                mlp_dim = opt.multimodal_attention_mlp_dim,
                                dropout = opt.multimodal_dropout,
                                emb_dropout = 0.1)
            edgetype_classifier.load_state_dict(torch.load('./saved_models/{}/edgetype_predictor_vg.pth'.format(opt.edge_pred_exp_name)))
        else:
            raise NotImplementedError('Use multimodal edge prediction')
        
        edgetype_classifier.eval()
        edgetype_classifier = edgetype_classifier.to(device)

    gsnn_annotation_dim = 2
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

    gsnn_net = gsnn.GSNN(opt, opt.state_dim, gsnn_annotation_dim, 
            graph.n_edge_types, opt.num_steps, opt.min_num_init, opt.context_dim, 
            opt.num_expand, opt.init_conf, graph.n_total_nodes, opt.node_bias_size, 
            opt.num_inter_steps, context_net_options, importance_net_options)

    node_name_list = makeLabelDict('nodename2index.txt')
    edgetype_name_list = makeLabelDict('edgetypename2index.txt')

    test_concept = opt.test_concepts
    if not opt.use_original_edges:
        graph.removeConnectionsNovelNode(test_concept)

    net = Classifier(opt, len(test_concept))

    if opt.load_net_type == 'VGG':
        encoder_net = VGG_Net()
    elif opt.load_net_type == 'ViT':
        from pytorch_pretrained_vit import ViT
        encoder_net = ViT('B_16_imagenet1k', pretrained=True)

    finetune_net = opt.finetune_net

    if finetune_net == 'classifier' or finetune_net == 'both':
        gsnn_net.importance_out_net.getNodeBiasNovel()
        gsnn_net.context_out_net.getNodeBiasNovel()

    net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/net.pth'))
    if opt.load_net_type == 'VGG':
        encoder_net.vgg_network_classifier.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/encoder_net_classifier.pth'))
    elif opt.load_net_type == 'ViT':
        encoder_net.fc.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/encoder_net_classifier.pth'))
    
    if finetune_net == 'gsnn' or finetune_net == 'seq' or finetune_net == 'classifier_only' or finetune_net == 'modified_gsnn':
        gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/gsnn_net.pth'))

    elif finetune_net == 'classifier':
        gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_gsnn.pth'))

    elif finetune_net == 'both':
        gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/novel_class_gsnn.pth'))

    ################ MAIN #################

    if opt.evaluate_finetune:
        novel_concept = ['bicycle', 'boat', 'stop sign', 'bird', 'backpack',
                    'frisbee', 'snowboard', 'surfboard', 'cup', 'fork', 'spoon', 'broccoli',
                    'chair', 'keyboard', 'microwave', 'vase']
        novel_concept = novel_concept[:opt.num_novel_nodes]
        nodes_removal_idx = [node_name_list.index(concept) for concept in novel_concept]

        num_novel_classes_trained = len(novel_concept)
        
        if opt.finetune_net == 'gsnn' or opt.finetune_net == 'both' or opt.finetune_net == 'seq':
            for i in range(len(novel_concept)):
                gsnn_net.importance_out_net.getNodeBiasNovel()
                gsnn_net.context_out_net.getNodeBiasNovel()
                gsnn_net.importance_out_net.updateNodeEmbedding()
                gsnn_net.context_out_net.updateNodeEmbedding()

            gsnn_net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/final_gsnn.pth'))
        
        net = Classifier(opt, 16 - num_novel_classes_trained)
        net.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/final_classifier.pth'))

        evaluate(opt=opt, encoder_net=encoder_net, gsnn_net=gsnn_net, classifier=net, graph=graph, 
                    novel_class_idx=nodes_removal_idx, curr_vocab_size=opt.vocab_size + num_novel_classes_trained)

    else:
        # novel_concept = ['bicycle', 'boat', 'stop sign', 'bird', 'backpack',
        #             'frisbee', 'snowboard', 'surfboard', 'cup', 'fork', 'spoon', 'broccoli',
        #             'chair', 'keyboard', 'microwave', 'vase']
        novel_concept = ['bicycle']

        nodes_removal_idx = []

        added_nodes = []
        prev_novel_images = []

        curr_vocab_size = opt.vocab_size - 1
        num_known_concepts = opt.vocab_size - len(novel_concept)

        embeddings_index = use_glove_embeddings()
        embedding_matrix_database, database_word_list = get_embedding_matrix_database(embeddings_index)
        num_nodes_added = 0

        for novel_single_concept in novel_concept:
            print ('Node: ', novel_single_concept)

            novel_single_concept = 'kitchen'# remove this later -- just for testing

            # Adding novel classes
            # Requires users to place novel images a directory called "novel_class_images_1", each image named: image_ID.jpg
            novel_image_names_full = glob.glob('./novel_class_images_1/{}/*'.format(novel_single_concept))
            novel_image_names = [img.split('/')[-1] for img in novel_image_names_full]
            novel_image_names_1 = ['./filtered_data_test/' + name.split('.')[0] + '.pth' for name in novel_image_names]
            img_list = []
            for file_name in novel_image_names_1:
                file_content = torch.load(file_name)
                name = file_content['name']
                img_path = name
                img_list.append(img_path)
            if opt.num_images_novel > 0:
                random.shuffle(img_list, random.random)
                img_list = img_list[:opt.num_images_novel]
            print ('Image List: ',img_list)
                        
            if not opt.use_original_edges:
                if finetune_net != 'modified_gsnn':
                    # We're sticking to concepts nodes for now, but for others we can also add a new input for type
                    # edges = getNodeConnections(opt, node_name=novel_single_concept, autoencoder=None, \
                    #                             edgetype_classifier=edgetype_classifier, graph=graph, device=device)
                    novel_concept_rep_idx = database_word_list.index(getNodeMapping(novel_single_concept))
                    novel_concept_rep = embedding_matrix_database[novel_concept_rep_idx]    

                    edges = novelClassEdgeAddition(opt, novel_concept_rep, num_known_concepts, img_list, graph, \
                                        gsnn_net, edgetype_classifier, encoder_net)

                    for edge in edges:
                        print (edge[0].name)
                    max_edge_addition = 10
                    if (len(edges) > max_edge_addition):
                        edges = edges[:max_edge_addition]
                    print ('Number of edges added to graph: ',len(edges))

            if finetune_net == 'gsnn' or finetune_net == 'seq' or finetune_net == 'classifier_only':
                if opt.use_original_edges:
                    neighbouring_idx_list = []
                    for edge in graph.edges:
                        if edge.start_node.name == novel_single_concept:
                            neighbouring_idx_list.append(edge.end_node.index)
                        elif edge.end_node.name == novel_single_concept:
                            neighbouring_idx_list.append(edge.start_node.index)
                else:
                    neighbouring_idx_list = [neighbour[0].index for neighbour in edges]

                # Initializing novel node bias, we only need to do that in the first step - GSNN pretraining
                gsnn_net.importance_out_net.getNodeBiasNovel(neighbouring_idx=neighbouring_idx_list)
                gsnn_net.context_out_net.getNodeBiasNovel(neighbouring_idx=neighbouring_idx_list)

            if finetune_net == 'modified_gsnn':
                gsnn_net.importance_out_net.getNodeBiasNovel()
                gsnn_net.context_out_net.getNodeBiasNovel()

            if not opt.use_original_edges:
                if finetune_net != 'modified_gsnn':
                    graph.updateGraphWithNovelNode(node_name=novel_single_concept, edges=edges, edgetype_name_list=edgetype_name_list)

            curr_vocab_size += 1

            gsnn_net, encoder_net, net, graph = trainSingleConcept(finetune_net, novel_img_list=img_list,
                                                gsnn_net=gsnn_net, encoder_net=encoder_net, net=net, graph=graph, 
                                                prev_images=prev_novel_images, curr_vocab_size=curr_vocab_size)
            added_nodes.append(novel_single_concept)
            prev_novel_images.append(img_list)

            num_known_concepts += 1
            num_nodes_added += 1

            if num_nodes_added == opt.num_novel_nodes:
                break

if (__name__ == '__main__'):
    main()
