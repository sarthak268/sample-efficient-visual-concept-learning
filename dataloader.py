import random
from itertools import cycle
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pickle
import os

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from utils.plot_utils import saveImage
from utils.data_utils import makeLabelDict
from graph.graph import Graph, saveObject

class data_graph(data.Dataset):
    def __init__(self, opt, data_dir, img_dir, get_detection=False, take_subset=False, subset_ratio=0.0, \
                    novel_class_images=[], test_train_split=False, is_train=True, nodes_to_remove_vocab=[], \
                    nodes_to_remove_detections=[], normalize=False):
        '''
        opt: argmument file
        data_dir: dir for storing all pth files
        img_dir: dir for storing all train images
        get_detection: get FasterRCNN detections 
        take_subset: take a subset of the dataset
        subset_ratio: what should be size of subset wrt total dataset
        novel_class_images: images that are from novel class --  basically used to augment dataset post pretraining
                            (give only when you want to evaluate on the novel image labels as well)
                            ------>>> Don't use since we assume we don't have access to previous data
        test_train_split: use test-train split
        is_train: use train split 
        nodes_to_remove_vocab: remove labels for certain nodes from GT (indices)
        nodes_to_remove_detections: remove labels for certain nodes from FasterRCNN detections (indices)
        '''
        super(data_graph, self).__init__()
        
        self.opt = opt
        self.take_subset = take_subset
        self.subset_ratio = subset_ratio
        self.all_files = glob.glob(data_dir + '/*pth')
        self.novel_class_images = novel_class_images
        if self.take_subset:
            random.shuffle(self.all_files)
            self.all_files = self.all_files[:int(len(self.all_files) * self.subset_ratio)]
        self.nodes_to_remove_vocab = nodes_to_remove_vocab
        self.nodes_to_remove_detections = nodes_to_remove_detections

        self.img_dir = img_dir
        self.normalize = normalize
        self.get_detection = get_detection

        print ('Dataset size original: ',len(self.all_files))

        self.test_train_split = test_train_split
        self.is_train = is_train

        if self.test_train_split:
            train_size = int(len(self.all_files) * 0.8)
            if self.is_train:
                self.all_files = self.all_files[:train_size]
            else:
                self.all_files = self.all_files[train_size:]

    def th_delete(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]

    # Converts a table of detections and sorts them into right form for annotation net
    def convertDetectionData(self, detections):
        num_det_class = self.opt.detector_size 
        detectConf = torch.zeros((num_det_class))
        
        for j in range(len(detections)):
            # Get detection data
            detection = detections[j]
            class_ind = detection['class_ind']
            conf = detection['conf']
            # hidden = detection['hidden']
            
            detectConf[class_ind - 1] = conf
        
        if len(self.nodes_to_remove_detections) > 0:
            detectConf[self.nodes_to_remove_detections] = 0.

        return detectConf.unsqueeze(-1)

    def getTrainExample(self, index, get_detection):
        file_name = self.all_files[index]

        file_content = torch.load(file_name)

        name = file_content['name']
        img_path = self.img_dir + name
            
        image = Image.open(img_path)
        if self.opt.load_net_type == 'VGG':
            image = image.resize((256, 256))
        elif self.opt.load_net_type == 'ViT':
            image = image.resize((384, 384))
        image = np.asarray(image).astype('float64')
        image_torch = torch.from_numpy(image).permute(2, 0, 1).float()

        label = file_content['present'].float()

        # This is for finetuning our models on entire dataset + novel class
        # Mostly not required
        if len(self.novel_class_images) > 0:
            # we assume only one novel class is present
            # for more than one novel class, we can add these list of images into a list
            # length of this list represents the number of classes added

            label = label.unsqueeze(-1)
            if img_path in novel_class_images:
                label[-1] = 1.

        if len(self.nodes_to_remove_vocab) > 0:
            label = self.th_delete(label, self.nodes_to_remove_vocab)

        if get_detection:
            detections = file_content['detections']

        if self.normalize:
            image_torch /= 255.

        if get_detection:
            return image_torch, detections, label
        else:
            return image_torch, label

    def __getitem__(self, index):
        if self.get_detection:
            img, detections, label = self.getTrainExample(index, get_detection=True)
        else:
            img, label = self.getTrainExample(index, get_detection=False)

        if self.get_detection:
            detectConf = self.convertDetectionData(detections) 
            return img, detectConf, label
        else:
            return img, label

    def __len__(self):
        return len(self.all_files)


if (__name__ == '__main__'):
    # dataset = data_baseline('./t7_data/', '/home/sarthak/data/VisualGenome/', False)
    # loader = cycle(DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True))

    # image, label, label_inds = next(loader)
    # print(image.shape, label.shape)

    from args.args_continual import opt
    from graph.graph import Graph, saveObject
    graph = Graph(detector_size=opt.detector_size, vocab_size=opt.vocab_size)
    graph = pickle.load(open('graph4_none.pkl', 'rb'))

    detector_reverse_lookup = list(graph.detector_reverse_lookup)

    dataset = data_graph(opt, './filtered_data/', '/data/sarthakbhagat/VisualGenome/', True)
    loader = cycle(DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True))

    image, detectConf, label = next(loader)
    saveImage(image[0])

    index2node = makeLabelDict('nodename2index.txt')

    # idx = torch.nonzero(detectConf.view(-1))
    # for i in idx:
    # 	print (index2node[i.item()])

    idx = torch.nonzero(label.view(-1))
    for i in idx:
    	print (index2node[i.item()])

    detect_idx = torch.nonzero(detectConf.view(-1))
    detect_l = [detector_reverse_lookup[idx] for idx in detect_idx]

    print ('===========')
                                    
    for i in detect_l:
    	print (index2node[i])