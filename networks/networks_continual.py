import torch
import torch.nn as nn
import torchvision.models.vgg as models

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# class EdgeTypeClassifier(torch.nn.Module):
#     def __init__(self, opt, in_dim, out_dim, normalize_pred=True):
#         super(EdgeTypeClassifier, self).__init__()

#         self.opt = opt
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.normalize_pred = normalize_pred
#         self.createEdgeTypeClassifier()

#     def createEdgeTypeClassifier(self, hidden_dim=64):
#         self.single_layer = nn.Linear(self.in_dim, self.out_dim)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()    

#     def forward(self, node1, node2):
#         x = torch.cat((node1, node2), dim=-1)
#         x = self.single_layer(x)
#         if self.normalize_pred:
#             x = self.sigmoid(x)
#         return x

