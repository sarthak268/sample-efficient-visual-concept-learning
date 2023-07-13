import torch
import copy
import math
import pickle
import numpy as np
import random

from utils.data_utils import isColor, makeLabelDict

def saveObject(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

class Node():
    def __init__(self, index, name, in_vocab, has_neil_detector, vocab_index, neil_detector_index, nodetype=None):
        
        # Index in graph
        self.index = index

        # Name of node
        self.name = name

        # Whether it is in vocabulary of captioning task (or other classification task)
        self.in_vocab = in_vocab

        # Whether this node has an associated NEIL detector
        self.has_neil_detector = has_neil_detector

        # Index into caption vocabulary. -1 if it's not in vocab
        assert((not in_vocab and vocab_index == -1) or (in_vocab and vocab_index >= 0))
        self.vocab_index = vocab_index

        if not has_neil_detector and neil_detector_index == -2:
            neil_detector_index = -1

        # Index into neil detectors. -1 if it's not a detector
        assert((not has_neil_detector and neil_detector_index == -1) or (has_neil_detector and neil_detector_index >= 0))
        self.neil_detector_index = neil_detector_index

        # Set up edge lists
        self.outgoing_edges = []
        self.incoming_edges = []

        self.nodetype = nodetype

class EdgeType():
    def __init__(self, index, name):
        self.index = index
        self.name = name

class Edge():
    def __init__(self, edgetype, start_node, end_node, index, confidence):
        self.edgetype = edgetype
        self.start_node = start_node
        self.end_node = end_node
        self.confidence = confidence
        self.index = index

class Graph():
    def __init__(self, detector_size, vocab_size):

        self.detector_size = detector_size
        self.vocab_size = vocab_size
        self.detector_reverse_lookup = torch.LongTensor(self.detector_size).fill_(-1)
        self.vocab_reverse_lookup = torch.LongTensor(self.vocab_size).fill_(-1)

        # Default is empty graph
        self.n_total_nodes = 0
        self.n_total_edges = 0  
        self.n_edge_types = 0
        self.nodes = []                # Table of nodes in the graph
        self.edges = []
        self.neil_detector_nodes = []  # Table of nodes that have NEIL detections
        self.vocab_nodes = []          # Table of nodes that are in vocab
        self.edge_types = []

    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def getNode2NodetypeMapping(self):
        self.attribute_nodes = makeLabelDict('nodetypes_attributes.txt')
        self.affordance_nodes = makeLabelDict('nodetypes_affordances.txt')
        self.concept_nodes = makeLabelDict('nodetypes_concepts.txt')
        self.all_nodes = makeLabelDict('nodename2index_corrected.txt')
        self.nodetype_mapping_list = []
        self.nodetype_list = []
        for i in range(len(self.all_nodes)):
            tensor = torch.zeros((3))
            if self.all_nodes[i] in self.attribute_nodes:
                tensor[0] = 1.
                nodetype = 'attribute'
            elif self.all_nodes[i] in self.affordance_nodes:
                tensor[1] = 1.
                nodetype = 'affordance'
            elif self.all_nodes[i] in self.concept_nodes:
                tensor[2] = 1.
                nodetype = 'concept'
            else:
                print (self.all_nodes[i])
                raise Exception('Node does not belong to any relevant type...')
            self.nodetype_mapping_list.append(tensor)
            self.nodetype_list.append(nodetype)

    def makeBirectional(self):
        for edge in self.edges:
            # add birectional edge only if the end node is not leaf
            if edge.end_node.name not in self.attribute_nodes or edge.end_node.name not in self.affordance_nodes:
                if not self.checkEdgeExists(edge.end_node.index, edge.start_node.index):
                    edge_type_idx = 0
                    edge_type_name = 'none'
                    start_node_idx = edge.end_node.index
                    end_node_idx = edge.start_node.index
                    confidence = edge.confidence
                    self.addEdge(edge_type_idx, edge_type_name, start_node_idx, end_node_idx, confidence)

    def cleanGraph(self, make_birectional=False):
        '''
        Ensure that all attribute and affordance nodes are leaf nodes
        '''
        for j, node in enumerate(self.nodes):
            if self.nodetype_list[node.index] == 'affordance' or self.nodetype_list[node.index] == 'attribute':
                outs = node.outgoing_edges
                for out in outs:
                    start_node = out.start_node
                    edgetype = out.edgetype 
                    end_node = out.end_node
                    for i, edge in enumerate(self.edges):
                        if edge.start_node.index == start_node.index and edge.end_node.index == end_node.index:
                            self.edges.pop(i)
                self.nodes[j].outgoing_edges = []

        if make_birectional:
            self.makeBirectional()

    def getNodeMask(self, nodetype):
        mask = torch.zeros((len(self.all_nodes)))
        indices = []
        for i, node in enumerate(self.nodetype_list):
            if node == nodetype:
                mask[i] = 1
                indices.append(i)
        return mask, indices
        
    def addNode(self, name, in_vocab, has_neil_detector, vocab_index, neil_detector_index):
        '''
        Add empty node to graph
        '''

        newnode = Node(self.n_total_nodes, name, in_vocab, has_neil_detector, vocab_index, neil_detector_index)

        # Update counts and lists 
        self.n_total_nodes += 1
        self.nodes.append(newnode)
        if newnode.in_vocab:
            self.vocab_nodes.append(newnode)
            self.vocab_reverse_lookup[vocab_index] = self.n_total_nodes
        
        if newnode.has_neil_detector:
            self.neil_detector_nodes.append(newnode)
            self.detector_reverse_lookup[neil_detector_index] = self.n_total_nodes

        # assuming that the node added is a concept node
        tensor = tensor = torch.zeros((3))
        tensor[2] = 1.
        nodetype = 'concept'
        self.nodetype_mapping_list.append(tensor)
        self.nodetype_list.append(nodetype)

    def addEdgeType(self, edge_type_idx, edge_type_name):
        '''
        Add edge types without adding edges
        '''

        assert(edge_type_idx == self.n_edge_types + 1)
        edgetype = EdgeType(edge_type_idx, edge_type_name)
        self.edge_types.append(edgetype)
        self.n_edge_types += 1 

    def addEdge(self, edge_type_idx, edge_type_name, start_node_idx, end_node_idx, confidence):
        '''
        Add new edge to graph
        '''

        if edge_type_idx > self.n_edge_types - 1:
            assert(edge_type_idx == self.n_edge_types)
            edgetype = EdgeType(edge_type_idx, edge_type_name)
            self.edge_types.append(edgetype)
            self.n_edge_types += 1 
        else:
            edgetype = EdgeType(0,'none')

        # Create edge
        startnode = self.nodes[start_node_idx]
        endnode = self.nodes[end_node_idx]
        edge = Edge(edgetype, startnode, endnode, self.n_total_edges + 1, confidence)
        startnode.outgoing_edges.append(edge)
        endnode.incoming_edges.append(edge)
        self.edges.append(edge)
        self.n_total_edges += 1

    def checkEdgeExists(self, start_node_idx, end_node_idx):
        '''
        Check if edge already exists
        '''

        edge_exists = False
        edge_idx = -1
        
        # Get start node
        startnode = self.nodes[start_node_idx]
        outgoing = startnode.outgoing_edges

        for i in range(len(outgoing)):
            edge = outgoing[i]
            if edge.start_node.index == start_node_idx and edge.end_node.index == end_node_idx:      
                edge_exists = True
                edge_idx = edge.index

        return edge_exists, edge_idx

    def getFullGraph(self):
        '''
        Get the edges of the entire graph
        '''

        edges = []
        for i, edge in enumerate(self.edges):
            e = [edge.start_node.index, edge.edgetype.index, edge.end_node.index]
            edges.append(e)

        return edges

    def removeEdge(self, start_node, end_node, edgetype):
        '''
        Removes given edge from graph
        '''
        self.n_total_edges -= 1.
        for edge_idx, edge in enumerate(self.edges):
            if start_node.index == edge.start_node.index and end_node.index == edge.end_node.index and \
                        edge.edgetype.index == edgetype:
                self.edges.pop(edge_idx)
                break
        
        for edge_idx, edge in enumerate(start_node.outgoing_edges):
            if edge.end_node.index == end_node.index and edge.edgetype.index == edgetype:
                start_node.outgoing_edges.pop(edge_idx)
                break

        for edge_idx, edge in enumerate(end_node.incoming_edges):
            if edge.start_node.index == start_node.index and edge.edgetype.index == edgetype:
                end_node.incoming_edges.pop(edge_idx)
                break   

    def removeNode(self, node_name):
        self.n_total_nodes -= 1
        for i, node in enumerate(self.nodes):
            if node.name == node_name:
                self.nodes.pop(i)
                for edge in self.edges:
                    if edge.start_node.name == node_name or \
                        edge.end_node.name == node_name:
                        self.removeEdge(edge.start_node, edge.end_node, edge.edgetype.index)
                break     

    def getExpandedGraph(self, reverse_lookup, active_idx, expand_idx, edges, edge_conf):
        '''
        Given reverse lookup table for graph, currently active nodes and list of nodes to expand, 
        return new active nodes, reverse lookup table, and edge list
        '''

        n_active = len(active_idx)

        for i in range(len(expand_idx)):
            toexp_idx = expand_idx[i]
            toexp_node = self.nodes[toexp_idx]

            # Look at all incoming and outgoing nodes
            for j in range(len(toexp_node.outgoing_edges)):
                cur_edge = toexp_node.outgoing_edges[j]
                cur_node = cur_edge.end_node
                cur_idx = cur_node.index
            
                # If not in graph, add to graph 
                if reverse_lookup[cur_idx] == -1:
                    active_idx.append(cur_idx)
                    n_active += 1
                    reverse_lookup[cur_idx] = n_active - 1

                    # Go through edges to new node and add them (if they are in our subgraph)
                    for k in range(len(cur_node.outgoing_edges)):
                        add_edge = cur_node.outgoing_edges[k]
                        other_node_idx = add_edge.end_node.index
                        
                        # If other end of edge is in the graph already, add the edge
                        if reverse_lookup[other_node_idx] != -1:
                            newedge = [reverse_lookup[cur_idx], add_edge.edgetype.index, reverse_lookup[other_node_idx]]
                            edges.append(newedge)
                            edge_conf.append(add_edge.confidence) 

                    for k in range(len(cur_node.incoming_edges)):
                        add_edge = cur_node.incoming_edges[k]
                        other_node_idx = add_edge.start_node.index
                        
                        # If other end of edge is in the graph already, add the edge
                        if reverse_lookup[other_node_idx] != -1:
                            newedge = [reverse_lookup[other_node_idx], add_edge.edgetype.index, reverse_lookup[cur_idx]]
                            edges.append(newedge)
                        edge_conf.append(add_edge.confidence)

                for j in range(len(toexp_node.incoming_edges)):
                    cur_node = toexp_node.incoming_edges[j].start_node
                    cur_idx = cur_node.index
                    if reverse_lookup[cur_idx] == -1:
                        active_idx.append(cur_idx)
                        n_active += 1
                        reverse_lookup[cur_idx] = n_active - 1                  
            
                        # Go through edges to new node and add them (if they are in our subgraph)
                        for k in range(len(cur_node.outgoing_edges)):
                            add_edge = cur_node.outgoing_edges[k]
                            other_node_idx = add_edge.end_node.index
                            
                            # If other end of edge is in the graph already, add the edge
                            if reverse_lookup[other_node_idx] != -1:
                                newedge = [reverse_lookup[cur_idx], add_edge.edgetype.index, reverse_lookup[other_node_idx]]
                                edges.append(newedge)
                                edge_conf.append(add_edge.confidence)
                            
                        for k in range(len(cur_node.incoming_edges)):
                            add_edge = cur_node.incoming_edges[k]
                            other_node_idx = add_edge.start_node.index
                            
                            # If other end of edge is in the graph already, add the edge
                            if reverse_lookup[other_node_idx] != -1:
                                newedge = [reverse_lookup[other_node_idx], add_edge.edgetype.index, reverse_lookup[cur_idx]]
                                edges.append(newedge)
                                edge_conf.append(add_edge.confidence)

        return reverse_lookup, active_idx, edges, edge_conf
        
    def getInitialNodesFromDetections(self, init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num):
        '''
        Get list of active nodes from detections
        '''

        initial_detections = copy.deepcopy(init_det_orig)

        above_threshold = torch.gt(initial_detections, conf_thresh).float()
        detect_inds = []
        annotations = []

        if above_threshold.sum().item() < min_num:
            for i in range(min_num):
                m = torch.max(initial_detections).item()
                j = torch.argmax(initial_detections).item()
                detect_inds.append(j)

                ann = []
                ann.append(m)
                for k in range(ann_total_size - 1):
                    ann.append(annotations_orig[j][k])
                annotations.append(ann)
                initial_detections[j] = -1

        else:
            for i in range(self.detector_size):
                if initial_detections[i] > conf_thresh:
                    detect_inds.append(i)
                    
                    ann = []
                    ann.append(initial_detections[i])
                    for k in range(ann_total_size - 1):
                        ann.append(annotations_orig[i][k])
                    annotations.append(ann)

        active_idx = []
        reverse_lookup = torch.LongTensor(self.n_total_nodes).fill_(-1)
        
        for i in range(len(detect_inds)):
            detect_ind = detect_inds[i]
            graph_ind = self.detector_reverse_lookup[detect_ind]
            assert(graph_ind != -1)
            active_idx.append(graph_ind.item())
            reverse_lookup[graph_ind] = i

        return active_idx, reverse_lookup, annotations 

    def getInitialGraph(self, init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num):
        '''
        Given the initial detections, give the graph with the initially detected classes and their neighbors
        '''

        active_idx, reverse_lookup, annotations = \
            self.getInitialNodesFromDetections(init_det_orig, annotations_orig, \
                                ann_total_size, conf_thresh, min_num)

        expand_idx = []
        for i in range(len(active_idx)):
            expand_idx.append(active_idx[i])

        edges = []
        edge_conf = []

        for i in range(len(active_idx)):
            cur_node_idx = active_idx[i]
            cur_node = self.nodes[cur_node_idx] 

            for j in range(len(cur_node.outgoing_edges)):
                cur_edge = cur_node.outgoing_edges[j]
                other_node_idx = cur_edge.end_node.index

                # If other end of edge is in graph, add edge
                if reverse_lookup[other_node_idx] != -1:
                    newedge = [reverse_lookup[cur_node_idx], cur_edge.edgetype.index, reverse_lookup[other_node_idx]] 
                    edges.append(newedge)
                    edge_conf.append(cur_edge.confidence) 

        reverse_lookup, active_idx, edges, edge_conf = \
                    self.getExpandedGraph(reverse_lookup, active_idx, expand_idx, edges, edge_conf)

        active_idx_types = []
        for i in range(len(active_idx)):
            active_idx_types.append(self.nodetype_mapping_list[active_idx[i]])
        
        return annotations, reverse_lookup, active_idx, expand_idx, edges, edge_conf, active_idx_types

    def updateGraphFromImportance(self, importance_orig, reverse_lookup, active_idx, expanded_idx, edges, edge_conf, num_expand):
        # Make copy of importance 
        importance = copy.deepcopy(importance_orig.detach())

        # First, minus out all of the already expanded nodes
        num_left = len(active_idx) - len(expanded_idx)
        if num_left == 0:
            # Nothing to do, all nodes expanded
            # DEBUG
            # print('All nodes expanded!')
            active_idx_types = []
            for i in range(len(active_idx)):
                active_idx_types.append(self.nodetype_mapping_list[active_idx[i]])
            return reverse_lookup, active_idx, expanded_idx, edges, edge_conf, active_idx_types

        for i in range(len(expanded_idx)):
            node_idx = expanded_idx[i]
            temp_idx = reverse_lookup[node_idx].item()
            importance[temp_idx] = -1
            
        # Get top important nodes and add to expand lists
        to_expand_idx = []
        for i in range(min(num_expand, num_left)):
            subgraph_idx = torch.argmax(importance).item()
            # Max importance
            real_idx = active_idx[subgraph_idx]
            expanded_idx.append(real_idx) 
            to_expand_idx.append(real_idx)
            importance[subgraph_idx] = -1
        
        # Now expand graph
        reverse_lookup, active_idx, edges, edge_conf = \
                self.getExpandedGraph(reverse_lookup, active_idx, to_expand_idx, edges, edge_conf)

        active_idx_types = []
        for i in range(len(active_idx)):
            active_idx_types.append(self.nodetype_mapping_list[active_idx[i]])
        
        return reverse_lookup, active_idx, expanded_idx, edges, edge_conf, active_idx_types

    # Given importance predictions, do graph update
    # This version uses importance input and just checks if each node should be expanded or not
    def updateGraphFromImportanceSelection(self, importance_orig, reverse_lookup, active_idx, expanded_idx, edges, edge_conf):
        # Make copy of importance 
        importance = copy.deepcopy(importance_orig)

        # First, minus out all of the already expanded nodes
        num_left = len(active_idx) - len(expanded_idx)
        if num_left == 0:
            # -- Nothing to do, all nodes expanded
            # -- DEBUG
            # --print('All nodes explanded!')
            return reverse_lookup, active_idx, expanded_idx, edges, edge_conf

        for i in range(len(expanded_idx)):
            node_idx = expanded_idx[i]
            temp_idx = reverse_lookup[node_idx]
            importance[temp_idx] = -1

        # Now select all the nodes above 0.5 (ones that have been selected)
        to_expand_idx = [] 
        for i in range(importance.shape[0]):
            if importance[i] > 0.5:
                real_idx = active_idx[i]
                expanded_idx.append(real_idx)
                to_expand_idx.append(real_idx)
            
        # Now expand graph
        reverse_lookup, active_idx, edges, edge_conf = \
                        self.getExpandedGraph(reverse_lookup, active_idx, to_expand_idx, edges, edge_conf)

        active_idx_types = []
        for i in range(len(active_idx)):
            active_idx_types.append(self.nodetype_mapping_list[active_idx[i]])
        
        return reverse_lookup, active_idx, expanded_idx, edges, edge_conf, active_idx_types

    # Given target nodes in that should be 1, propogates value to all nodes in graph
    # Returns value for each node
    def getDiscountedValues(self, vocab_target_idx, gamma, num_steps):
        node_values = torch.zeros((self.n_total_nodes))
        visited = torch.zeros((self.n_total_nodes))

        # First get graph indices and set targets to 1
        frontier = []
        for i in range(len(vocab_target_idx)):
            cur_vocab_idx = vocab_target_idx[i]
            graph_idx = self.vocab_reverse_lookup[cur_vocab_idx]
            if graph_idx != -1:
                node_values[graph_idx] = 1
                visited[graph_idx] = 1
                frontier.append(graph_idx)
            
        # Loop over steps and set discounted rewards
        value = 1
        
        # Loop over steps
        for step in range(num_steps):
            value = value * gamma
            new_frontier = []
        
            # Look at nodes on frontier, and set values for their neighbors
            for i in range(len(frontier)):
                front_node_idx = frontier[i]
                front_node = self.nodes[front_node_idx]
                for j in range(len(front_node.outgoing_edges)):
                    edge = front_node.outgoing_edges[j]
                    other_node_idx = edge.end_node.index 
                    if visited[other_node_idx] == 0:
                        assert(node_values[other_node_idx] == 0)
                        new_frontier.append(other_node_idx)
                        node_values[other_node_idx] = value
                        visited[other_node_idx] = 1
                    
                for j in range(len(front_node.incoming_edges)): 
                    edge = front_node.incoming_edges[j]
                    other_node_idx = edge.start_node.index
                    if visited[other_node_idx] == 0:
                        assert(node_values[other_node_idx] == 0)
                        new_frontier.append(other_node_idx)
                        node_values[other_node_idx] = value
                        visited[other_node_idx] = 1
                    
            frontier = new_frontier

            # If all nodes have been visited, or the nodes are not discoverable
            # then we're done
            if len(frontier) == 0:
                break
            
        return node_values

    def updateGraphEdgeAddition(self, dropout_prob=0.1):
        '''
        Remove random nodes from the graph
        '''
        removal_mask = np.where(np.random.random(self.n_total_nodes) < dropout_prob, 1, 0)
        nodes_removal_idx = np.argwhere(removal_mask == 1).reshape(-1)
        nodes_to_remove = [self.nodes[idx] for idx in nodes_removal_idx]
        print ('Number of removed nodes: ',len(nodes_to_remove))
        nodes_idx_to_remove = [self.index for nodes in nodes_to_remove]
        for edge in self.edges:
            if edge.start_node.index in nodes_idx_to_remove or edge.end_node.index in nodes_idx_to_remove:
                self.removeEdge(edge.start_node, edge.end_node, edge.edgetype.index)

    def updateGraphEdgeAdditionStratifiedSampling(self, dropout_prob=0.1):
        '''
        Remove random edges from the graph in a stratified manner
        '''
        edges_to_remove = []
        for edgetype_idx in range(self.n_edge_types):
            curr_edgetype = []
            for edge in self.edges:
                if edge.edgetype.index == edgetype_idx:
                    curr_edgetype.append(edge)
            current_remove = random.sample(curr_edgetype, int(len(curr_edgetype) * dropout_prob))
            edges_to_remove.extend(current_remove)
        print ('Number of removed edges: {} out of a total of: {}'.format(len(edges_to_remove), len(self.edges)))
        for edge in edges_to_remove:
            self.removeEdge(edge.start_node, edge.end_node, edge.edgetype.index)

    def updateGraphWithNovelNode(self, node_name, edges, edgetype_name_list):
        self.addNode(node_name, in_vocab=False, has_neil_detector=False, vocab_index=-1, neil_detector_index=-1)
        novel_node = self.nodes[-1]

        for edge in edges:
            graph_node, edge_type_idx, edge_dir, confidence = edge
            edge_type_name = edgetype_name_list[edge_type_idx]

            if edge_dir == 'in':
                start_node = graph_node
                end_node = novel_node
            elif edge_dir == 'out':
                start_node = novel_node
                end_node = graph_node
            else:
                raise Exception('Invalid edge_dir')
            
            start_node_idx = start_node.index
            end_node_idx = end_node.index
            self.addEdge(edge_type_idx, edge_type_name, start_node_idx, end_node_idx, confidence)

    def checkNodesNotPresent(self):
        node_present = []
        for edge in self.edges:
            if edge.start_node.index not in node_present:
                node_present.append(edge.start_node.index)
            if edge.end_node.index not in node_present:
                node_present.append(edge.end_node.index)
        not_present = []
        for idx in range(316):
            if idx not in node_present:
                not_present.append(idx)
        node_name_list = makeLabelDict('nodename2index.txt')
        not_present_classes = [node_name_list[idx] for idx in not_present]

    def removeConnectionsNovelNode(self, node_name):
        not_present_nodes = ['player', 'edge', 'lady', 'hill', 'skier', 'windshield', 'back', 'parking meter', \
                            'wine glass', 'hot dog', 'potted plant', 'toaster', 'scissors', 'hair drier', \
                            'shiny', 'thin', 'thick', 'present', 'lit', 'stacked', 'light brown']
        if node_name not in not_present_nodes:
            for node in self.nodes:
                if node.name == node_name:
                    node.name += '-removed'
            for edge in self.edges:
                if edge.start_node.name in node_name or edge.end_node.name in node_name:
                    print ('Removing edge b/w {} and {}'.format(edge.start_node.name, edge.end_node.name))
                    self.removeEdge(edge.start_node, edge.end_node, edge.edgetype.index)
            