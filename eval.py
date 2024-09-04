import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import cosine_similarity

from triplet import TripletNetwork
from libs.dataset import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v',
        '--val_path',
        type=str,
        help="Path to directory containing validation dataset.",
        required=True
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for saving prediction images.",
        required=True
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # Установка устройства на CUDA, если доступно, иначе CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_dataset = Dataset(args.val_path, shuffle_triplets=False, augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TripletNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    losses = []
    correct = 0
    total = 0

    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    
    for i, ((img1, img2, img3), y, (class1, class2, class3)) in enumerate(val_dataloader):
        print("[{} / {}]".format(i, len(val_dataloader)))

        img1, img2, img3 = map(lambda x: x.to(device), [img1, img2, img3])
        class1 = class1[0]
        class2 = class2[0]
        class3 = class3[0]

        anchor = model(img1)
        positive = model(img2)
        negative = model(img3)

        # Вычисление косинусного сходства
        pos_sim = cosine_similarity(anchor, positive)
        neg_sim = cosine_similarity(anchor, negative)
        
        # Классификация по косинусному сходству
        correct += (pos_sim > neg_sim).sum().item()
        total += len(pos_sim)

        # Отображение изображений и их предсказаний
        fig = plt.figure(figsize=(14, 4))

        # Применение обратного преобразования (денормализация) для восстановления оригинальных изображений
        img1 = inv_transform(img1).cpu().numpy()[0]
        img2 = inv_transform(img2).cpu().numpy()[0]
        img3 = inv_transform(img3).cpu().numpy()[0]

        # Отображение первого изображения
        ax = fig.add_subplot(1, 3, 1)
        img1 = np.moveaxis(img1, 0, -1)
        img1 = np.clip(img1, 0, 1)
        plt.imshow(img1)
        plt.axis("off")
        plt.title(f"Anchor:\n{class1}")

        # Отображение второго изображения
        ax = fig.add_subplot(1, 3, 2)
        img2 = np.moveaxis(img2, 0, -1)
        img2 = np.clip(img2, 0, 1)
        plt.imshow(img2)
        plt.axis("off")
        plt.title(f"Similarity={pos_sim[0].item():.2f}\n{class2}")

        # Отображение третьего изображения
        ax = fig.add_subplot(1, 3, 3)
        img3 = np.moveaxis(img3, 0, -1)
        img3 = np.clip(img3, 0, 1)
        plt.imshow(img3)
        plt.axis("off")
        plt.title(f"Similarity={neg_sim[0].item():.2f}\n{class3}")

        # Сохранение изображения
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_path, '{}.png').format(i))

    accuracy = correct / total
    print("Validation: Accuracy={:.2f}".format(accuracy))
