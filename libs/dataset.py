import os
import glob
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset as TorchDataset
import torch.nn.functional as F


class Dataset(TorchDataset):
    def __init__(self, path, shuffle_triplets=True, augment=False):
        self.path = path
        self.shuffle_triplets = shuffle_triplets
        self.augment = augment

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop((224, 320), pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop((224, 320), pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.create_triplets()
        self.class_names = sorted(list(set(os.path.relpath(path, self.path).split(os.sep)[0] for path in self.image_paths)))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def create_triplets(self):
        self.image_paths = []
        self.make_model_year_indices = {}
        self.make_model_year_paths = {}

        for ext in ['*.jpg', '*.png']:
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith(tuple(ext)):
                        path = os.path.join(root, file)
                        make_model_year = os.path.relpath(root, self.path)
                        if make_model_year not in self.make_model_year_indices:
                            self.make_model_year_indices[make_model_year] = []
                            self.make_model_year_paths[make_model_year] = []
                        self.make_model_year_indices[make_model_year].append(len(self.image_paths))
                        self.make_model_year_paths[make_model_year].append(path)
                        self.image_paths.append(path)

        self.indices1 = np.arange(len(self.image_paths))

        if self.shuffle_triplets:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            np.random.seed(1)

        self.indices2 = []
        self.indices3 = []

        for idx in self.indices1:
            anchor_path = self.image_paths[idx]
            make_model_year = os.path.relpath(os.path.dirname(anchor_path), self.path)

            # Positive example (same make/model/year)
            positive_paths = self.make_model_year_paths[make_model_year]
            if len(positive_paths) > 1:
                pos_idx = np.random.choice([i for i in self.make_model_year_indices[make_model_year] if i != idx])
                self.indices2.append(pos_idx)
            else:
                self.indices2.append(idx)  # Handle case where no positive example is available

            # Negative example (same make but different model/year)
            possible_negative_make_model_years = [k for k in self.make_model_year_paths.keys() if k.startswith(make_model_year.split('/')[0])]
            possible_negative_make_model_years.remove(make_model_year)
            if possible_negative_make_model_years:
                neg_make_model_year = np.random.choice(possible_negative_make_model_years)
                neg_paths = self.make_model_year_paths[neg_make_model_year]
                neg_idx = np.random.choice(self.make_model_year_indices[neg_make_model_year])
                self.indices3.append(neg_idx)
            else:
                self.indices3.append(idx)  # Handle case where no negative example is available

        self.indices2 = np.array(self.indices2)
        self.indices3 = np.array(self.indices3)

    def __getitem__(self, index):
        idx, idx2, idx3 = self.indices1[index], self.indices2[index], self.indices3[index]
        img1 = Image.open(self.image_paths[idx]).convert("RGB")
        img2 = Image.open(self.image_paths[idx2]).convert("RGB")
        img3 = Image.open(self.image_paths[idx3]).convert("RGB")

        if self.transform:
            img1 = self.transform(img1).float()
            img2 = self.transform(img2).float()
            img3 = self.transform(img3).float()

        make = os.path.relpath(os.path.dirname(self.image_paths[idx]), self.path)
        make = make.split('/')[0]  # Extract only the make
        make_idx = self.class_to_idx[make]
        
        return img1, img2, img3, make_idx

    def __len__(self):
        return len(self.indices1)

    def num_classes(self):
        return len(self.class_names)
