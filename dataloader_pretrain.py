import os
import random
import glob
import numpy as np
from PIL import Image
from typing import List
from itertools import cycle
# import clip

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from detector.fasterrcnn_coco import detector
from utils.data_utils import makeLabelDict

class data_pretrain(data.Dataset):
    def __init__(self, opt, data_dir, img_dir, novel_node_img, novel_class_ratio=0.5, \
                subset_classes=[], novel_class_idx=[], novel_class_idx_detect=[], prev_images=[]):
        '''
        opt: arguments
        data_dir: data directory with all train pth files 
        img_dir: directory with all images 
        novel_node_img: all images with label of current novel class 
        novel_class_ratio: ratio of images with current novel class in each batch 
        subset_classes: select only classes with these subsets 
        novel_class_idx: vocab idx of all novel classes 
        novel_class_idx_detect: detection idx of all novel classes 
        prev_images: list of lists containing images from novel classes on which we have already trained
        '''

        super(data_pretrain, self).__init__()
        
        self.opt = opt
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.novel_class_ratio = novel_class_ratio
        self.novel_node_img = novel_node_img
        self.all_files = glob.glob(data_dir + '/*pth')
        self.subset_classes = subset_classes
        self.novel_class_idx = novel_class_idx
        self.novel_class_idx_detect = novel_class_idx_detect
        self.prev_images = prev_images

        if len(self.subset_classes) > 0:
            all_files_new = []
            for index in range(len(self.all_files)):
                file_name = self.all_files[index]
                file_content = torch.load(file_name)
                label = file_content['present'].float()
                if torch.any(label[subset_classes] == 1.):
                    all_files_new.append(file_name)
            self.all_files = all_files_new
            print ('Number of samples with selected classes: ', len(self.all_files))

    def th_delete(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]

    # Converts a table of detections and sorts them into right form for annotation net
    def convertDetectionData(self, detections):
        num_det_class = self.opt.detector_size 
        detect_conf = torch.zeros((num_det_class))
        
        for j in range(len(detections)):
            # Get detection data
            detection = detections[j]
            class_ind = detection['class_ind']
            conf = detection['conf']
            
            detect_conf[class_ind - 1] = conf

        if len(self.novel_class_idx_detect) > 0:
            detect_conf[self.novel_class_idx_detect] = 0.
        
        return detect_conf.unsqueeze(-1)

    def getTrainExample(self, index):
        file_name = self.all_files[index]
        file_content = torch.load(file_name)
        name = file_content['name']
        
        # Making sure we do not include any novel image data 
        # Wouldn't need this
        while (name in self.novel_node_img):
            index = random.randint(0, len(self.all_files) - 1)
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

        if len(self.novel_class_idx) > 0:
            label = self.th_delete(label, self.novel_class_idx)

        detections = file_content['detections']

        if len(self.prev_images) > 0:
            for novel_img_idx in range(len(self.prev_images)):
                if file_name in self.prev_images[novel_img_idx]:
                    last_tensor = torch.tensor([1.])
                else:
                    last_tensor = torch.tensor([0.])
                label = torch.cat((label, last_tensor), dim=-1)
            
        return image_torch, detections, label
        
    def getNovelClassExample(self, image_path, augment_image=True):
        img_name = image_path.split('/')[-1].split('.')[0]
        file_name = self.data_dir + 'data_' + img_name + '.pth'
        file_content = torch.load(file_name)

        img_path = self.img_dir + image_path
        image = Image.open(img_path)
        if self.opt.load_net_type == 'VGG':
            image = image.resize((256, 256))
        elif self.opt.load_net_type == 'ViT':
            image = image.resize((384, 384))

        if augment_image:
            augment = random.random()
            if augment < 0.9:
                # Random transformations -- hyperparameter, change and see what works!
                trans = transforms.RandomApply(
                    torch.nn.ModuleList([transforms.RandomAffine(degrees=45, \
                                            translate=(0.1, 0.1), scale=(2, 2))]), p=0.9)
                image = trans(image)
            
        image = np.asarray(image).astype('float64')
        image_torch = torch.from_numpy(image).permute(2, 0, 1).float()
            
        label = file_content['present'].float()

        if len(self.novel_class_idx) > 0:
            label = self.th_delete(label, self.novel_class_idx)

        detections = file_content['detections']

        if len(self.prev_images) > 0:
            for novel_img_idx in range(len(self.prev_images)):
                if file_name in self.prev_images[novel_img_idx]:
                    last_tensor = torch.tensor([1.])
                else:
                    last_tensor = torch.tensor([0.])
                label = torch.cat((label, last_tensor), dim=-1)

        return image_torch, detections, label
        
    def __getitem__(self, index):
        prob = random.random()
        if prob < self.novel_class_ratio:
            is_novel_class = True
        else:
            is_novel_class = False

        if is_novel_class:
            if isinstance(self.novel_node_img, List):
                # Case where we are given multiple images 
                # of the same novel class selected by the user
                image_name = random.choice(self.novel_node_img) 
            else:
                raise Exception('Not implemented')
            img, detections, label = self.getNovelClassExample(image_name)
        else:
            img, detections, label = self.getTrainExample(index)

        detect_conf = self.convertDetectionData(detections) 

        return img, detect_conf, label, is_novel_class

    def __len__(self):
        return self.opt.pretrain_dataset_size

class data_pretrain_evaluation(data.Dataset):
    def __init__(self, opt, data_dir, img_dir, novel_class_idx, test_classes_idx_detect, remaining_class_idx, \
                    only_novel_class, subset_classes=[]):
        super(data_pretrain_evaluation, self).__init__()
        
        self.opt = opt
        self.data_dir = data_dir # for test
        self.img_dir = img_dir
        self.novel_class_idx = novel_class_idx
        self.test_classes_idx_detect = test_classes_idx_detect
        self.remaining_class_idx = remaining_class_idx
        self.only_novel_class = only_novel_class
        self.all_data_files = glob.glob(data_dir + '/*pth')
        self.all_data_files = self.all_data_files[int(len(self.all_data_files) * 0.8):]

        if len(subset_classes) > 0:
            self.all_files = []
            for file in self.all_data_files:
                file_content = torch.load(file)
                label = file_content['present'].float()
                if torch.any(label[subset_classes] > 0):
                    self.all_files.append(file)
        else:
            if self.only_novel_class:
                # just making sure, ideally not required
                self.all_files = []
                for file in self.all_data_files:
                    file_content = torch.load(file)
                    label = file_content['present'].float()
                    if torch.any(label[novel_class_idx] > 0):
                        self.all_files.append(file)
            else:
                self.all_files = self.all_data_files
        print ('Dataset size ===', len(self.all_files))

    def th_delete(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]

    # Converts a table of detections and sorts them 
    # into right form for annotation net
    def convertDetectionData(self, detections):
        num_det_class = self.opt.detector_size 
        detect_conf = torch.zeros((num_det_class))
        
        for j in range(len(detections)):
            # Get detection data
            detection = detections[j]
            class_ind = detection['class_ind']
            conf = detection['conf']
            
            detect_conf[class_ind - 1] = conf

        detect_conf[self.test_classes_idx_detect] = 0.
        
        return detect_conf.unsqueeze(-1)

    def getTrainExample(self, index):
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

        if len(self.remaining_class_idx) > 0:
            label = self.th_delete(label, self.remaining_class_idx)

        detections = file_content['detections']
            
        return image_torch, detections, label
        
    def __getitem__(self, index):
        img, detections, label = self.getTrainExample(index)
        detect_conf = self.convertDetectionData(detections) 
        return img, detect_conf, label

    def __len__(self):
        return len(self.all_files)

if (__name__ == '__main__'):
    from args.args_continual import opt

    # novel_node_name = 'toothbrush'
    # novel_node_image = ['VG_100K/2374251.jpg',
    #             'VG_100K/2342469.jpg',
    #             'VG_100K_2/2391390.jpg',
    #             'VG_100K/2354378.jpg',
    #             'VG_100K/2315417.jpg',
    #             'VG_100K/2361505.jpg',
    #             'VG_100K/2335079.jpg',
    #             'VG_100K/2365289.jpg',
    #             'VG_100K/2350710.jpg',
    #             'VG_100K/2347497.jpg',
    #             'VG_100K_2/2409096.jpg']

    # dataset = data_pretrain(opt, './filtered_data/', '/home/sarthak/data/VisualGenome/', novel_node_image, novel_class_ratio=1.)
    # loader = cycle(DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True))
    # img, detectConf, detectClass, label, is_novel_class = next(loader)
    # print (img.shape, detectConf.shape, detectClass.shape, label.shape, is_novel_class)
