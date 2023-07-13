import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import os
import random
import pickle
from PIL import Image
from tensorboardX import SummaryWriter

from utils.plot_utils import saveImage
from args.args_continual import opt
from embeddings.glove_embeddings import use_glove_embeddings, get_embedding_matrix
from utils.data_utils import makeLabelDict, getNodeMapping
from transformer.relate import EdgeTransformer

class data_graph_nodes(data.Dataset):
    def __init__(self, data_path, img_dir, is_train=True, train_ratio=0.8, add_no_connection=True, no_connection_ratio=1.):

        super(data_graph_nodes, self).__init__()
        self.data_path = data_path
        self.img_dir = img_dir 
        self.is_train = is_train
        self.add_no_connection = add_no_connection
        self.no_connection_ratio = no_connection_ratio
                
        with open('image_data_glove.pkl', 'rb') as fp:
            image_data = pickle.load(fp)

        with open('all_relationship_glove.pkl', 'rb') as fp:
            relationship_data = pickle.load(fp)

        print ('Loaded data...')

        if self.add_no_connection:
            node_name_list = makeLabelDict('nodename2index_corrected.txt') 
            embeddings_index = use_glove_embeddings()
            embedding_matrix_vocab = get_embedding_matrix(node_name_list, embeddings_index)

            for idx in range(len(relationship_data)):
                sub_vec, sub, edgetype_vec, obj_vec, obj = relationship_data[idx]
                img = image_data[idx]
                
                new_node = random.choice(node_name_list)
                new_node_vocab_idx = node_name_list.index(new_node)
                new_node_rep = embedding_matrix_vocab[new_node_vocab_idx]
                
                if random.random() < 0.5: 
                    rel = 'new_sub'  
                else:
                    rel = 'new_obj'

                edgetype_vec = np.array([0])
                
                if rel == 'new_sub':  
                    new_data = [new_node_rep, new_node, edgetype_vec, obj_vec, obj]
                elif rel == 'new_obj':
                    new_data = [sub_vec, sub, edgetype_vec, new_node_rep, new_node]
                relationship_data.append(new_data)
                image_data.append(img)

        if self.is_train:
            self.relationship_data = relationship_data[:int(len(relationship_data)*train_ratio)]
            self.image_data = image_data[:int(len(image_data)*train_ratio)]
        else:
            self.relationship_data = relationship_data[int(len(relationship_data)*train_ratio):]
            self.image_data = image_data[int(len(image_data)*train_ratio):]
        
    def __getitem__(self, index):
        img_idx = self.image_data[index]
        sub_vec, sub, edgetype_vec, obj_vec, obj = self.relationship_data[index]

        file_name = self.data_path + 'data_' + str(img_idx) + '.pth'
        file_content = torch.load(file_name)

        name = file_content['name']
        img_path = self.img_dir + name

        image = Image.open(img_path)
        image = image.resize((256, 256))
        image = np.asarray(image).astype('float64')
        image_torch = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return  image_torch, sub_vec, torch.from_numpy(edgetype_vec), obj_vec
        
    def __len__(self):
        return len(self.relationship_data)    
        
def train():
    
    dataset = data_graph_nodes( './filtered_data/', opt.dataset_path, is_train=True, add_no_connection=opt.add_no_connection)
    dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=0, drop_last=True)

    print ('Data loaded...')

    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(opt.seed)

    criterion = nn.BCELoss()
    criterion = criterion.to(device)

    num_classes = 1

    predictor = EdgeTransformer(image_size = 256,
                                patch_size = 32,
                                num_classes = 1,
                                dim = opt.multimodal_attention_dim,
                                word_dim=opt.node_embedding_dim,
                                depth = opt.multimodal_attention_depth,
                                heads = opt.multimodal_attention_num_heads,
                                mlp_dim = opt.multimodal_attention_mlp_dim,
                                dropout = opt.multimodal_dropout,
                                emb_dropout = 0.1)
    predictor = predictor.to(device)

    if opt.load_nets:
        predictor.load_state_dict(torch.load('./saved_models/' + opt.load_exp_name + '/edgetype_predictor_vg.pth'))

    optimizer = optim.Adam(predictor.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    if not os.path.exists('saved_models/' + opt.edge_pred_exp_name):
        os.makedirs('saved_models/' + opt.edge_pred_exp_name)

    writer = SummaryWriter('runs/' + opt.edge_pred_exp_name)

    for epoch in range(opt.num_epochs):
        total_loss = 0.0
            
        for iteration, batch in enumerate(dataloader):

            predictor.train()
            optimizer.zero_grad()

            image, node1_rep, edgetype_rep, node2_rep = batch
            node1_rep = node1_rep.to(device).float()
            node2_rep = node2_rep.to(device).float()
            edgetype_rep = edgetype_rep.to(device).float()
            image = image.to(device)

            output = predictor(image, node1_rep, node2_rep)
        
            total_loss = criterion(output, edgetype_rep) 

            total_loss.backward()
            optimizer.step()

            if iteration % opt.print_after == 0:
                print ('Epoch: [{} / {}], Iteration: [{} / {}], Total Loss: {}'.format(
                        epoch, opt.num_epochs, iteration, len(dataset) // opt.batchsize, round(total_loss.item(), 6)
                    ))

            if iteration % opt.plot_after == 0:
                # Write to tensorboard
                writer.add_scalar('Edge prediction loss', total_loss.item(),
                                epoch * (int(len(dataset) / opt.batchsize) + 1) + iteration)

        if epoch % opt.save_after == 0:
            torch.save(predictor.state_dict(), './saved_models/' + opt.edge_pred_exp_name + '/edgetype_predictor_vg.pth')

       

if (__name__ == '__main__'):
    train()
