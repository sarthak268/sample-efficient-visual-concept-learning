import torch
import torch.nn as nn

class UpdateNet(nn.Module):

    def __init__(self, opt, n_edge_types, state_dim):
    
        super(UpdateNet, self).__init__()

        self.opt = opt
        self.n_edge_types = n_edge_types
        self.state_dim = state_dim

        self.sigmoid = nn.Sigmoid()
        if opt.image_conditioned_propnet or opt.image_conditioned_propnet1:
            self.linear_node_update_gate = nn.Linear(self.state_dim * 8, self.state_dim * 2)
        else:
            self.linear_node_update_gate = nn.Linear(self.state_dim * 3, self.state_dim * 2)

        self.tanh = nn.Tanh()
        if opt.image_conditioned_propnet:
            self.linear_node_update_transform = nn.Linear(self.state_dim * 8, self.state_dim)
        else:
            self.linear_node_update_transform = nn.Linear(self.state_dim * 3, self.state_dim)

        if self.opt.image_conditioned_propnet or self.opt.image_conditioned_propnet1:
            self.representation_size = 4096 if opt.load_net_type == 'VGG' else 1000
            self.linear_embedding_layer = nn.Linear(self.representation_size, 5 * self.state_dim)

    def forward(self, edge_type_input, node_rep, adj_mats, image_embedding, n_nodes_list):
        
        n_graphs = adj_mats.shape[0]

        temp_input = edge_type_input[:self.n_edge_types, :, :].view(-1, self.state_dim)
        narrowed_input = adj_mats[0][:, :n_nodes_list[0] * self.n_edge_types]
        forward_input = torch.matmul(narrowed_input, temp_input)

        temp_input = edge_type_input[self.n_edge_types:, :, :].view(-1, self.state_dim)
        narrowed_input = adj_mats[0][:, n_nodes_list[0] * self.n_edge_types :]
        reverse_input = torch.matmul(narrowed_input, temp_input)

        current_state = node_rep
        
        if self.opt.image_conditioned_propnet or self.opt.image_conditioned_propnet1:
            # print (image_embedding.shape)
            repeated_image_embedding = image_embedding.unsqueeze(0).repeat(len(forward_input), 1)
            image_embedding_projection = self.linear_embedding_layer(repeated_image_embedding)
            joined_input = torch.cat((forward_input, reverse_input, current_state, image_embedding_projection), axis=-1)
        else:
            joined_input = torch.cat((forward_input, reverse_input, current_state), axis=-1)

        gates = self.sigmoid(self.linear_node_update_gate(joined_input))

        update_gate = gates[:, :self.state_dim]
        reset_gate = gates[:, self.state_dim:]

        if self.opt.image_conditioned_propnet:
            joined_input = torch.cat((forward_input, reverse_input, torch.multiply(reset_gate, current_state), \
                                            image_embedding_projection), axis=-1)
        else:
            joined_input = torch.cat((forward_input, reverse_input, torch.multiply(reset_gate, current_state)), axis=-1)

        transformed_output = self.tanh(self.linear_node_update_transform(joined_input))

        output = current_state + (update_gate * (transformed_output - current_state))
        output = output.unsqueeze(0)

        return output
