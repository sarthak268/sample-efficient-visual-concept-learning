import random
from itertools import cycle
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pickle
import json
import os

import torch
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from utils.plot_utils import saveImage
from utils.data_utils import makeLabelDict
from graph.graph import Graph, saveObject
from detector.fasterrcnn_coco import detector

class data_coco(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, opt, root, annFile, num_categories, transform=None, target_transform=None, remove_classes=[]):
        from pycocotools.coco import COCO
        self.root = root
        self.opt = opt
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.predictor = detector(threshold = .3, model_path = None)
        self.node_name_list = makeLabelDict('nodename2index.txt')
        self.num_nodes = len(self.node_name_list)
        self.num_categories = num_categories
        self.cats = self.coco.cats
        self.coco_name_list = makeLabelDict('mscoco_gt_idx2name.txt')
        self.remove_classes = remove_classes

    def th_delete(self, tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        
        label_one_hot = torch.zeros((self.num_nodes))
        for i in range(len(target)):
            category_names = self.cats[target[i]['category_id']]['name']
            idx = self.node_name_list.index(category_names)
            # idx = target[i]['category_id']
            label_one_hot[idx] = 1.

        if len(self.remove_classes) > 0:
            remove_classes_idx = [self.node_name_list.index(cls) for cls in self.remove_classes]
            label_one_hot = self.th_delete(label_one_hot, remove_classes_idx)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = np.asarray(img)
        if self.transform is not None:
            img = self.transform(img)

        img = Image.fromarray(img)

        if self.opt.load_net_type == 'VGG':
            img = img.resize((256, 256))
        elif self.opt.load_net_type == 'ViT':
            img = img.resize((384, 384))
        img = np.asarray(img)
        img_torch = torch.from_numpy(img).float().permute(2, 0, 1)

        boxes, pred_cls = self.predictor.detect(img_torch)

        detects_one_hot = torch.zeros((self.num_categories))
        for i in range(len(pred_cls)):
            # idx = self.node_name_list.index(pred_cls[i])
            idx = self.coco_name_list.index(pred_cls[i])
            detects_one_hot[idx] = 1.

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_torch, label_one_hot, detects_one_hot

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if (__name__ == '__main__'):
    from args.args_continual import opt
    dataset_type = 'val'
    root_dir = '../../pytorch-faster-rcnn/COCODevKit/{}2017/'.format(dataset_type)
    annFile = '../../pytorch-faster-rcnn/COCODevKit/annotations/instances_{}2017.json'.format(dataset_type)
    dataset = data_coco(opt, root=root_dir, annFile=annFile, num_categories=opt.detector_size)
    # loader = cycle(DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True))
    # image, labels, labels_one_hot, detects, detects_one_hot = next(loader)
    # print (image.shape, labels_one_hot.shape, detects_one_hot.shape)