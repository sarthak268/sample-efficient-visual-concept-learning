import torch
import torch.nn as nn
from ggnn.ggnn_networks import UpdateNet

class BaseGGNN(nn.Module):

    def __init__(self, opt, n_steps, n_edge_types):

        super(BaseGGNN, self).__init__()

        self.opt = opt
        self.device = torch.device(self.opt.device) if torch.cuda.is_available() else torch.device("cpu")
        self.n_steps = n_steps
        self.state_dim = self.opt.state_dim
        self.n_edge_types = n_edge_types
                
        self.createOneNodeUpdateNet()
        self.createPropagationNetModules()
        self.constructNetworkForGraphs()

    def createPropagationNetModules(self):
        self.linear_prop_net_layers = []
        self.linear_reverse_prop_net_layers = []
        for edge_type in range(self.n_edge_types):
            layer_in_dim = self.state_dim
            layer_out_dim = self.state_dim
            linear_prop_net = nn.Linear(layer_in_dim, layer_out_dim)
            self.linear_prop_net_layers.append(linear_prop_net)
            linear_reverse_prop_net = nn.Linear(layer_in_dim, layer_out_dim)
            self.linear_reverse_prop_net_layers.append(linear_reverse_prop_net)
        
    def createOneNodeUpdateNet(self):
        self.add_net = UpdateNet(self.opt, self.n_edge_types, self.state_dim)
        self.add_net = self.add_net.to(self.device)

    def constructNetworkForGraphs(self):                
        self.prop_net = [ [] for _ in range(2 * self.n_edge_types) ]
        for e_type in range(self.n_edge_types):
            self.prop_net[e_type] = self.linear_prop_net_layers[e_type] 
            self.prop_net[e_type + self.n_edge_types] = self.linear_reverse_prop_net_layers[e_type]
        self.prop_net = nn.ModuleList(self.prop_net)

        self.prop_net = self.prop_net.to(self.device)

    def forwardWithAdjacencyMatricesAndConcatenatedAnnotationMatrix(self, adjacency_list, annotations, image_embedding=None):        
        n_nodes_list = []
        n_total_nodes = 0
        for i, adj in enumerate(adjacency_list):
            n_nodes_list.append(adj.shape[0])
            n_total_nodes += adj.shape[0]
        
        self.n_nodes_list = n_nodes_list

        self.a_list = adjacency_list
        self.n_graphs = len(self.a_list)

        self.prop_inputs = []

        assert(annotations.shape[0] == n_total_nodes)

        input1 = torch.zeros(n_total_nodes, self.state_dim).to(self.device)
        
        input1[:, :annotations.shape[1]] = annotations
        
        self.prop_inputs.append(input1)

        for i_step in range(self.n_steps):
        
            edge_type_list = torch.zeros((2 * self.n_edge_types, n_total_nodes, self.state_dim)).to(self.device)
            
            for i in range (self.n_edge_types * 2):
                edge_type_list[i, :, :] = self.prop_net[i](self.prop_inputs[i_step])
            
            node_rep = self.prop_inputs[i_step]
            adj_mats_list = self.a_list[0].unsqueeze(0)

            self.prop_inputs.append(self.add_net(edge_type_list, node_rep, adj_mats_list, image_embedding, self.n_nodes_list))

        return self.prop_inputs[-1]

    def forward(self, edges_list, annotations_list, image_embedding=None):
        
        if torch.is_tensor(annotations_list):   # single tensor case
            assert(torch.is_tensor(edges_list[0]))  # edges_list must be a list of adjacency matrices
            return self.forwardWithAdjacencyMatricesAndConcatenatedAnnotationMatrix(edges_list, annotations_list, image_embedding=image_embedding)

        assert(len(edges_list) == len(annotations_list))

        annotations_tensor_list = []
        adjacency_matrix_list = []

        for i, annotations in enumerate(annotations_list):
            if not torch.is_tensor(annotations):
                annotations_tensor_list.append(torch.from_numpy(np.asarray(annotations)))
            else:
                annotations_tensor_list.append(annotations)

            if not torch.is_tensor(edges_list[i]):
                adjacency_matrix_list.append(ggnn.create_adjacency_matrix_cat(edges_list[i], 
                                                        annotations_tensor_list[i].shape[0], self.n_edge_types))
            else:
                adjacency_matrix_list.append(edges_list[i])

        return self.forward_with_adjacency_and_annotation_matrices(adjacency_matrix_list, self.n_steps, annotations_tensor_list)