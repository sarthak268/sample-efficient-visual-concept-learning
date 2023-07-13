import torch
import numpy as numpy
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import cv2

def saveGraphNodes(final_label, detected_classes, confidences, active_idx, expanded_idx, initial_conf,
                        initial_nodes, index2node, save_dir, exp_name, epoch=-1, iteration=-1):

    active_nodes = [index2node[i] for i in active_idx]
    # initial_detected_nodes = [index2node[i] for i in detected_classes]
    initial_detected_nodes = [index2node[i] for i in initial_nodes]
    final_label_idx = torch.nonzero(final_label)
    ground_truth_labels = [index2node[i] for i in final_label_idx]
    expanded_nodes = [index2node[i] for i in expanded_idx]
    file = open("{}/{}/node_expansion_data.txt".format(save_dir, exp_name), "a")
    if iteration > 0:
        file.writelines('Epoch : {}, Iteration: {}'.format(epoch, iteration) + '\n')
    file.writelines('Initially detected nodes: ' + ','.join(initial_detected_nodes) + '\n')
    file.writelines('Active Nodes: ' + ','.join(active_nodes) + '\n')
    file.writelines('Expanded Nodes: ' + ','.join(expanded_nodes) + '\n')
    file.writelines('Ground Truth Classes: ' + ','.join(ground_truth_labels) + '\n')
    file.writelines('\n')

def saveImage(im, image_name='sample_image.png'):
    img_data = im.permute(1, 2, 0).numpy()
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_name, img_data)
