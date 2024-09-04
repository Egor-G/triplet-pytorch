import os
import glob
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle_triplets=True, augment=False):
        '''
        Create an iterable dataset from a directory containing sub-directories of 
        entities with their images contained inside each sub-directory.

            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_triplets (boolean): Pass True when training, False otherwise. When set to false, the image triplet generation will be deterministic
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each triplet of images
        '''
        self.path = path

        self.feed_shape = [3, 224, 224]
        self.shuffle_triplets = shuffle_triplets

        self.augment = augment

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(self.feed_shape[1:])
            ])

        self.create_triplets()

    def create_triplets(self):
        '''
        Creates lists of indices that will form the triplets, to be fed for training or evaluation.
        '''

        self.image_paths = list(Path(self.path).rglob("*.[jp][pn]g"))
        self.image_classes = []
        self.class_indices = {}

        for image_path in self.image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            self.class_indices[image_class].append(self.image_paths.index(image_path))

        self.indices1 = np.arange(len(self.image_paths))

        if self.shuffle_triplets:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            np.random.seed(1)

        self.indices2 = []
        self.indices3 = []

        for idx in self.indices1:
            class1 = self.image_classes[idx]

            # Positive example (same class)
            pos_class = class1
            pos_idx = np.random.choice(self.class_indices[pos_class])
            self.indices2.append(pos_idx)

            # Negative example (different class)
            neg_class = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
            neg_idx = np.random.choice(self.class_indices[neg_class])
            self.indices3.append(neg_idx)

        self.indices2 = np.array(self.indices2)
        self.indices3 = np.array(self.indices3)

    def __iter__(self):
        self.create_triplets()

        for idx, idx2, idx3 in zip(self.indices1, self.indices2, self.indices3):
            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]
            image_path3 = self.image_paths[idx3]

            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")
            image3 = Image.open(image_path3).convert("RGB")

            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()
                image3 = self.transform(image3).float()

            yield (image1, image2, image3), torch.FloatTensor([1]), (self.image_classes[idx], self.image_classes[idx2], self.image_classes[idx3])

    def __len__(self):
        return len(self.image_paths)
