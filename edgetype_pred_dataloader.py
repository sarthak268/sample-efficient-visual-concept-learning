import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import random
import pickle
from PIL import Image
import numpy as np

from utils.data_utils import makeLabelDict, getNodeMapping
from embeddings.glove_embeddings import use_glove_embeddings, get_embedding_matrix
from embeddings.gpt_embeddings import vocab_embeddings

class data_graph_nodes(data.Dataset):
    def __init__(self, data_path, img_dir, is_train=True, train_ratio=0.8, add_no_connection=True, no_connection_ratio=1.):

        super(data_graph_nodes, self).__init__()
        self.data_path = data_path
        self.img_dir = img_dir 
        self.is_train = is_train
        self.add_no_connection = add_no_connection
        self.no_connection_ratio = no_connection_ratio

        with open('saved_data/image_cond_relation/image_data_gpt.pkl', 'rb') as fp:
            image_data = pickle.load(fp)

        with open('saved_data/image_cond_relation/all_relationship_gpt.pkl', 'rb') as fp:
            relationship_data = pickle.load(fp)

        print ('loaded data...')

        if self.add_no_connection:
            node_name_list = makeLabelDict('nodename2index_corrected.txt') 
            embedding_matrix_vocab = vocab_embeddings()

            for idx in range(len(relationship_data)):
                sub_vec, edgetype_vec, obj_vec = relationship_data[idx]
                img = image_data[idx]
                
                new_node = random.choice(node_name_list)
                new_node_vocab_idx = node_name_list.index(new_node)
                new_node_rep = embedding_matrix_vocab[new_node_vocab_idx]
                new_node_rep = new_node_rep.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                new_node_rep = new_node_rep[-1]
                
                if random.random() < 0.5: 
                    rel = 'new_sub'  
                else:
                    rel = 'new_obj'

                edgetype_vec = np.array([0])
                
                if rel == 'new_sub':  
                    new_data = [new_node_rep, edgetype_vec, obj_vec]

                elif rel == 'new_obj':
                    new_data = [sub_vec, edgetype_vec, new_node_rep]

                relationship_data.append(new_data)
                image_data.append(img)

            print (len(relationship_data))

        if self.is_train:
            self.relationship_data = relationship_data[:int(len(relationship_data)*train_ratio)]
            self.image_data = image_data[:int(len(image_data)*train_ratio)]
        else:
            self.relationship_data = relationship_data[int(len(relationship_data)*train_ratio):]
            self.image_data = image_data[int(len(image_data)*train_ratio):]

    def __getitem__(self, index):
        img_idx = self.image_data[index]
        sub_vec, edgetype_vec, obj_vec = self.relationship_data[index]

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

if (__name__ == '__main__'):
    dataset = data_graph_nodes( './filtered_data/', '/home/sarthak/data/VisualGenome/', is_train=True, add_no_connection=True, no_connection_ratio=1.)