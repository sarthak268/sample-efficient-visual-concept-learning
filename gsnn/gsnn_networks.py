import torch
import torch.nn as nn

class ContextNet(nn.Module):
    def __init__(self, opt, n_total_nodes):
        super(ContextNet, self).__init__()

        self.opt = opt
        in_dim = self.opt.state_dim
        self.n_total_nodes = n_total_nodes

        self.sigmoid = nn.Sigmoid()

        if self.opt.context_use_node_input:
            self.embedding_node_bias = nn.Embedding(self.n_total_nodes, self.opt.node_bias_size)
            in_dim += self.opt.node_bias_size

        if self.opt.context_use_ann:
            annotation_dim = 2  # Hack - annotation after conf is just 0's
            in_dim += annotation_dim

        if self.opt.use_nodetypes:
            in_dim += 3

        if self.opt.context_architecture == 'gated':
            self.linear_gate = nn.Linear(in_dim, self.opt.context_dim)

        if self.opt.context_architecture == 'tanhout' :
            self.tanh = self.Tanh()

        self.layer_input_linear_list = []
        h_dim = self.opt.context_out_net_h_size
        if self.opt.context_out_net_h_size > 0:
            for i in range(self.opt.context_out_net_num_layer): 
                layer_input_linear = nn.Linear(in_dim, h_dim)
                self.layer_input_linear_list.append(layer_input_linear)
                in_dim = h_dim
            self.layer_input_linear_list = nn.ModuleList(self.layer_input_linear_list)

        if self.opt.context_transfer_function == 'tanh': 
            self.layer_input_act = nn.Tanh()        
        elif self.opt.context_transfer_function == 'sigmoid': 
            self.layer_input_act = nn.Sigmoid()
        elif self.opt.context_transfer_function == 'relu' :
            self.layer_input_act = nn.ReLU()
        else:
            raise Exception('Option ' + self.opt.context_transfer_function + ' not valid')

        self.output_linear = nn.Linear(in_dim, self.opt.context_dim)

        self.total_nodes_w_embedding = self.n_total_nodes

    def getNodeBiasNovel(self, neighbouring_idx=[]):
        self.embedding_node_bias_novel = nn.Embedding(1, self.opt.node_bias_size)
            
        if len(neighbouring_idx) > 0:
            avg_embedding = torch.zeros((self.opt.node_bias_size)).to(self.opt.device)
            for neighbour_idx in neighbouring_idx:
                avg_embedding += self.embedding_node_bias.weight[neighbour_idx].to(self.opt.device)
            avg_embedding /= len(neighbouring_idx)
            self.embedding_node_bias_novel.weight = nn.Parameter(avg_embedding.unsqueeze(0))

    def updateNodeEmbedding(self):
        if self.opt.context_use_node_input:
            new_embedding_node_bias = nn.Embedding(self.total_nodes_w_embedding + 1, self.opt.node_bias_size)
            with torch.no_grad():
                new_embedding_node_bias.weight[:-1, :] = self.embedding_node_bias.weight
                new_embedding_node_bias.weight[-1, :] = self.embedding_node_bias_novel.weight
            self.embedding_node_bias = new_embedding_node_bias
            self.total_nodes_w_embedding += 1

    def forward(self, x, node_input=None, ann_input=None, nodetype_input=None):
        joined_input = x

        # Go through all possible input combos
        if self.opt.context_use_node_input:
            # node_bias = self.embedding_node_bias(node_input.long())
            node_bias = torch.zeros((x.shape[0], self.opt.node_bias_size)).to(x.device) 
            node_input = node_input.long()
            for i in range(len(node_input)):
                node_bias[i, :] = self.embedding_node_bias_novel(torch.tensor([0]).to(x.device)) \
                                        if node_input[i] >= self.opt.vocab_size \
                                        else self.embedding_node_bias(node_input[i])

            joined_input = torch.cat((joined_input, node_bias), axis=-1)

        if self.opt.context_use_ann: 
            joined_input = torch.cat((joined_input, ann_input), axis=-1)

        if self.opt.use_nodetypes:
            joined_input = torch.cat((joined_input, nodetype_input), axis=-1)

        if self.opt.context_architecture == 'gated':
            gate = self.sigmoid(self.linear_gate(joined_input))

        layer_input = joined_input 
        for layer_input_linear in self.layer_input_linear_list:
            layer_input = self.layer_input_act(layer_input_linear(layer_input))
        
        output = self.output_linear(layer_input)

        if self.opt.context_architecture == 'gated':
            final_output = output * gate
        elif self.opt.context_architecture == 'linout': 
            final_output = output
        elif self.opt.context_architecture == 'sigout' :
            final_output = self.sigmoid(output)
        elif self.opt.context_architecture == 'tanhout':
            final_output = 0.5 * (self.tanh(output) + 1)
        else:
            raise Exception('Option ' + self.opt.context_architecture + ' not valid')

        return final_output

class ImportanceNet(nn.Module):
    def __init__(self, opt, n_total_nodes):
        super(ImportanceNet, self).__init__()

        self.opt = opt
        in_dim = self.opt.state_dim
        self.n_total_nodes = n_total_nodes

        self.sigmoid = nn.Sigmoid()

        if self.opt.importance_use_node_input:
            self.embedding_node_bias = nn.Embedding(self.n_total_nodes, self.opt.node_bias_size)
            in_dim += self.opt.node_bias_size

        if opt.importance_use_ann:
            annotation_dim = 2  # Hack - annotation after conf is just 0's
            in_dim += annotation_dim

        if opt.use_nodetypes:
            in_dim += 3

        if self.opt.importance_architecture == 'gated' or self.opt.importance_architecture == 'gatedsig':
            gate_linear = nn.Linear(in_dim, 1)

        if self.opt.importance_architecture == 'tanhout' :
            self.tanh = self.Tanh()

        self.layer_input_linear_list = []
        h_dim = self.opt.importance_out_net_h_size
        if self.opt.importance_out_net_h_size > 0:
            for i in range(self.opt.importance_out_net_num_layer): 
                layer_input_linear = nn.Linear(in_dim, h_dim)
                self.layer_input_linear_list.append(layer_input_linear)
                in_dim = h_dim
            self.layer_input_linear_list = nn.ModuleList(self.layer_input_linear_list)

        if self.opt.importance_transfer_function == 'tanh': 
            self.layer_input_act = nn.Tanh()        
        elif self.opt.importance_transfer_function == 'sigmoid': 
            self.layer_input_act = nn.Sigmoid()
        elif self.opt.importance_transfer_function == 'relu' :
            self.layer_input_act = nn.ReLU()
        else:
            raise Exception('Option ' + self.opt.importance_transfer_function + ' not valid')

        self.output_linear = nn.Linear(in_dim, 1)

        self.total_nodes_w_embedding = self.n_total_nodes

    def getNodeBiasNovel(self, neighbouring_idx=[]):
        self.embedding_node_bias_novel = nn.Embedding(1, self.opt.node_bias_size)
            
        if len(neighbouring_idx) > 0:
            avg_embedding = torch.zeros((self.opt.node_bias_size)).to(self.opt.device)
            for neighbour_idx in neighbouring_idx:
                avg_embedding += self.embedding_node_bias.weight[neighbour_idx].to(self.opt.device)
            avg_embedding /= len(neighbouring_idx)
            self.embedding_node_bias_novel.weight = nn.Parameter(avg_embedding.unsqueeze(0))

    def updateNodeEmbedding(self):
        if self.opt.importance_use_node_input:
            new_embedding_node_bias = nn.Embedding(self.total_nodes_w_embedding + 1, self.opt.node_bias_size)
            with torch.no_grad():
                new_embedding_node_bias.weight[:-1, :] = self.embedding_node_bias.weight
                new_embedding_node_bias.weight[-1, :] = self.embedding_node_bias_novel.weight
            self.embedding_node_bias = new_embedding_node_bias
            self.total_nodes_w_embedding += 1

    def forward(self, x, node_input=None, ann_input=None, nodetype_input=None):
        joined_input = x

        # Go through all possible input combos
        if self.opt.importance_use_node_input:
            node_bias = torch.zeros((x.shape[0], self.opt.node_bias_size)).to(x.device) 
            node_input = node_input.long()
            for i in range(len(node_input)):
                node_bias[i, :] = self.embedding_node_bias_novel(torch.tensor([0]).to(x.device)) \
                                        if node_input[i] >= self.opt.vocab_size \
                                        else self.embedding_node_bias(node_input[i])

            joined_input = torch.cat((joined_input, node_bias), axis=-1)

        if self.opt.importance_use_ann: 
            joined_input = torch.cat((joined_input, ann_input), axis=-1)
        
        if self.opt.use_nodetypes:
            joined_input = torch.cat((joined_input, nodetype_input), axis=-1)

        if self.opt.importance_architecture == 'gated' or self.opt.importance_architecture == 'gatedsig':
            gate = self.sigmoid(self.linear_gate(joined_input))

        layer_input = joined_input
        for layer_input_linear in self.layer_input_linear_list:
            layer_input = self.layer_input_act(layer_input_linear(layer_input))
    
        output = self.output_linear(layer_input)

        if self.opt.importance_architecture == 'gated':
            final_output = output * gate
        elif self.opt.importance_architecture == 'gatedsig':
            final_output = self.sigmoid(output * gate)
        elif self.opt.importance_architecture == 'linout': 
            final_output = output
        elif self.opt.importance_architecture == 'sigout' :
            final_output = self.sigmoid(output)
        elif self.opt.importance_architecture == 'tanhout':
            final_output = 0.5 * (self.tanh(output) + 1)
        else:
            raise Exception('Option ' + self.opt.importance_architecture + ' not valid')

        return final_output