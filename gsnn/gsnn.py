import torch
import torch.nn as nn
import copy

from ggnn.baseggnn import BaseGGNN
from ggnn.ggnn_utils import createAdjacencyMatrixCat
from gsnn.gsnn_networks import ImportanceNet, ContextNet
from gsnn.gsnn_utils import createAnnTensorFromTable, getLookuptableRep

class GSNN(nn.Module):

    def __init__(self, opt, state_dim, annotation_dim, n_edge_types, num_steps, 
        min_num_init, context_dim, num_expand, init_conf, n_total_nodes, node_bias_size, 
        num_inter_steps, context_net_options, importance_net_options):

        super(GSNN, self).__init__()

        self.opt = opt
        self.device = torch.device(self.opt.device) if torch.cuda.is_available() else torch.device("cpu")
        self.state_dim = state_dim
        self.annotation_dim = annotation_dim  

        self.n_edge_types = n_edge_types
        self.num_steps = num_steps
        self.min_num_init = min_num_init
        self.context_dim = context_dim
        self.num_expand = num_expand
        self.init_conf = init_conf
        self.n_total_nodes = n_total_nodes
        self.node_bias_size = node_bias_size
        self.num_inter_steps = num_inter_steps
        
        # Output net variables
        self.context_architecture = context_net_options['architecture']
        self.context_transfer_function = context_net_options['transfer_function']
        self.context_use_node_input = context_net_options['use_node_input']
        self.context_use_annotation_input = context_net_options['use_annotation_input']
        self.importance_architecture = importance_net_options['architecture']
        self.importance_transfer_function = importance_net_options['transfer_function']
        self.importance_use_node_input = importance_net_options['use_node_input']
        self.importance_use_annotation_input = importance_net_options['use_annotation_input']
        self.importance_expand_type = importance_net_options['expand_type'] or "value"
    
        # Do some input checking
        if self.context_use_node_input == 1 and self.n_total_nodes <= 0:
            raise Exception('Need number of nodes to have node biases')  
        if self.importance_use_node_input == 1 and self.n_total_nodes <= 0:
            raise Exception('Need number of nodes to have node biases')

        self.createBaseNets()
        self.createOutputNets()

    def createBaseNets(self):
        self.baseggnns = BaseGGNN(self.opt, self.num_inter_steps, self.n_edge_types)
        self.baseggnns = self.baseggnns.to(self.device)

    def createOutputNets(self):
        self.context_out_net = ContextNet(self.opt, self.n_total_nodes)
        self.importance_out_net = ImportanceNet(self.opt, self.n_total_nodes)

    def forward(self, full_graph, initial_detections, initial_annotations, evaluation=False, 
                    pretrain_gsnn=False, curr_vocab_size=-1, image_embedding=None):

        annotations, reverse_lookup, active_idx, expanded_idx, edges, edge_conf, active_idx_types = \
                        full_graph.getInitialGraph(initial_detections, initial_annotations, 
                                                    self.annotation_dim, self.init_conf, 
                                                    self.min_num_init)

        self.initial_annotations_from_detections = annotations

        annotation_tensor = createAnnTensorFromTable(annotations, len(annotations), self.annotation_dim, self.device) 
        
        input_tensor = torch.zeros((len(active_idx), self.annotation_dim)).to(self.device)
        input_tensor[:len(annotations), :] = annotation_tensor

        if pretrain_gsnn:
            novel_node_ann = torch.tensor([curr_vocab_size, 0]).unsqueeze(0).to(self.device)
            input_tensor = torch.cat((input_tensor, novel_node_ann), dim=0)

            active_idx.append(curr_vocab_size)

            for j, edge in enumerate(full_graph.edges):
                if edge.start_node.index == curr_vocab_size:
                    newedge = [reverse_lookup[edge.start_node.index], edge.edgetype.index, \
                                    reverse_lookup[edge.end_node.index]]
                    edges.append(newedge)
                                                
        # Convert everything to tensors
        adjacency_matrix_list = []
        adjacency_matrix_list.append(createAdjacencyMatrixCat(edges, len(active_idx), self.n_edge_types, self.device))

        # State values
        self.past_active_idx = []
        initial_active_idx = []
        
        for aidx in range(len(active_idx)):
            initial_active_idx.append(active_idx[aidx])
        self.past_active_idx.append(initial_active_idx)

        self.past_expanded_idx = []
        self.initial_expanded_idx = []
        for eidx in range(len(expanded_idx)):
            self.initial_expanded_idx.append(expanded_idx[eidx])
        self.past_expanded_idx.append(self.initial_expanded_idx)

        self.base_net_inputs = []
        self.base_net_outputs = []
        importance_outputs = []
        self.base_num_nodes = []
        self.node_inputs = []
        self.ann_inputs = []
        self.active_idx_types_torch = []
        self.base_net_inputs.append(input_tensor)
        self.base_num_nodes.append(len(active_idx))

        for i in range(self.num_steps):
            # Forward through base ggnns
            # takes edge list, n steps, annotations list

            if self.opt.image_conditioned_propnet or self.opt.image_conditioned_propnet1:
                self.base_net_outputs.append(self.baseggnns(adjacency_matrix_list, self.base_net_inputs[i], 
                                                                image_embedding=image_embedding))
            else:
                self.base_net_outputs.append(self.baseggnns(adjacency_matrix_list, self.base_net_inputs[i]))

            if i != self.num_steps - 1:
                # Get annotation input
                zero_pad_ann_input = torch.zeros((len(active_idx), self.annotation_dim)).to(self.device)
                
                zero_pad_ann_input[:len(annotations), :] = annotation_tensor
                
                self.ann_inputs.append(zero_pad_ann_input)

                # Do importance calculation
                self.node_inputs.append(getLookuptableRep(active_idx, self.device))
                
                if self.opt.use_nodetypes:
                    self.active_idx_types_torch.append(torch.stack(active_idx_types).to(self.device))

                    if self.importance_use_node_input and self.importance_use_annotation_input:
                        importance = self.importance_out_net(self.base_net_outputs[i].squeeze(0), self.node_inputs[i], self.ann_inputs[i], \
                                                                nodetype_input=self.active_idx_types_torch[i])
        
                    elif self.importance_use_node_input and not self.importance_use_annotation_input:
                        importance = self.importance_out_net(self.base_net_outputs[i].squeeze(0), self.node_inputs[i], \
                                                                nodetype_input=self.active_idx_types_torch[i])
        
                    elif not self.importance_use_node_input and self.importance_use_annotation_input:
                        importance = self.importance_out_net(self.base_net_outputs[i].squeeze(0), self.ann_inputs[i], \
                                                                nodetype_input=self.active_idx_types_torch[i]) 

                    else:
                        importance = self.importance_out_net(self.base_net_outputs[i].squeeze(0), nodetype_input=self.active_idx_types_torch[i])

                else:
                    if self.importance_use_node_input and self.importance_use_annotation_input:
                        importance = self.importance_out_net(self.base_net_outputs[i].squeeze(0), self.node_inputs[i], self.ann_inputs[i])
        
                    elif self.importance_use_node_input and not self.importance_use_annotation_input:
                        importance = self.importance_out_net(self.base_net_outputs[i].squeeze(0), self.node_inputs[i])
        
                    elif not self.importance_use_node_input and self.importance_use_annotation_input:
                        importance = self.importance_out_net(self.base_net_outputs[i].squeeze(0), self.ann_inputs[i]) 

                    else:
                        importance = self.importance_out_net(self.base_net_outputs[i].squeeze(0))
                
                importance_outputs.append(importance)

                # Update graph information
                if self.importance_expand_type == "value": 
                    reverse_lookup, active_idx, expanded_idx, edges, edge_conf, active_idx_types = \
                                            full_graph.updateGraphFromImportance(importance.view(-1), 
                                                                                    reverse_lookup, active_idx, expanded_idx, edges, 
                                                                                    edge_conf, self.num_expand)
                elif self.importance_expand_type == "select":
                    reverse_lookup, active_idx, expanded_idx, edges, edge_conf, active_idx_types = \
                                            full_graph.updateGraphFromImportanceSelection(importance.view(-1), 
                                                                                            reverse_lookup, active_idx, expanded_idx, 
                                                                                            edges, edge_conf)                
                # Update in state
                new_active_idx = []
                for aidx in range(len(active_idx)):
                    new_active_idx.append(active_idx[aidx])
                self.past_active_idx.append(new_active_idx)

                new_expanded_idx = []
                for eidx in range(len(expanded_idx)):
                    new_expanded_idx.append(expanded_idx[eidx])
                self.past_expanded_idx.append(new_expanded_idx)

                # Update tensors and save everything
                adjacency_matrix_list = []
                adjacency_matrix_list.append(createAdjacencyMatrixCat(edges, len(active_idx), self.n_edge_types, self.device))
                
                next_input_tensor = torch.zeros((len(active_idx), self.state_dim)).to(self.device)
                
                next_input_tensor[:self.base_num_nodes[i], :] = self.base_net_outputs[i]

                self.base_net_inputs.append(next_input_tensor)
                self.base_num_nodes.append(len(active_idx))

        # Get context output
        self.node_inputs.append(getLookuptableRep(active_idx, self.device))

        zero_pad_ann_input = torch.zeros((len(active_idx), self.annotation_dim)).to(self.device)
        
        zero_pad_ann_input[:len(annotations), :] = annotation_tensor
        self.ann_inputs.append(zero_pad_ann_input)

        if self.opt.use_nodetypes:        
            self.active_idx_types_torch.append(torch.stack(active_idx_types).to(self.device))
            if (self.context_use_node_input and self.context_use_annotation_input):
                context_output = self.context_out_net(self.base_net_outputs[-1].squeeze(0), self.node_inputs[-1], \
                                                        self.ann_inputs[-1], nodetype_input=self.active_idx_types_torch[-1])
            elif self.context_use_node_input and not self.context_use_annotation_input:
                context_output = self.context_out_net(self.base_net_outputs[-1].squeeze(0), self.node_inputs[-1], \
                                                        nodetype_input=self.active_idx_types_torch[-1])
            elif not self.context_use_node_input and self.context_use_annotation_input:
                context_output = self.context_out_net(self.base_net_outputs[-1].squeeze(0), self.ann_inputs[-1], \
                                                        nodetype_input=self.active_idx_types_torch[-1])
            else:
                context_output = self.context_out_net(self.base_net_outputs[-1].squeeze(0), \
                                                        nodetype_input=self.active_idx_types_torch[-1])
        else:
            if (self.context_use_node_input and self.context_use_annotation_input):
                context_output = self.context_out_net(self.base_net_outputs[-1].squeeze(0), self.node_inputs[-1], \
                                                        self.ann_inputs[-1])
            elif self.context_use_node_input and not self.context_use_annotation_input:
                context_output = self.context_out_net(self.base_net_outputs[-1].squeeze(0), self.node_inputs[-1])
            elif not self.context_use_node_input and self.context_use_annotation_input:
                context_output = self.context_out_net(self.base_net_outputs[-1].squeeze(0), self.ann_inputs[-1])
            else:
                context_output = self.context_out_net(self.base_net_outputs[-1].squeeze(0))

        # Save some values we might want on hand
        self.reverse_lookup = reverse_lookup
        self.active_idx = active_idx
        self.expanded_idx = expanded_idx

        if not evaluation:
            return context_output, importance_outputs, reverse_lookup, active_idx, expanded_idx
        else:
            return context_output, importance_outputs, reverse_lookup, active_idx, expanded_idx, self.initial_expanded_idx, edges