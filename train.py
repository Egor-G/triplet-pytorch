import os
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from triplet import TripletNetwork
from libs.dataset import Dataset


def custom_collate(batch):
    """
    Custom collate function to handle triplet batches (anchor, positive, negative).
    Pads images to the size of the largest width and height in the batch.

    Args:
        batch (list of tuples): Each element in the batch is a tuple of (img1, img2, img3)
    
    Returns:
        Tuple of tensors: (batch_img1, batch_img2, batch_img3), where each is a batched tensor
        with images padded to the size of the largest image in the batch.
    """

    # Unzip the batch into three separate lists of images
    img1_list, img2_list, img3_list = zip(*batch)

    # Find the maximum width and height in the batch
    max_width = max([img.size(2) for img in img1_list + img2_list + img3_list])  # width = size(2)
    max_height = max([img.size(1) for img in img1_list + img2_list + img3_list]) # height = size(1)
    print(f"Padding to {max_width}x{max_height}")

    def pad_image(image, max_height, max_width):
        """
        Pads an image to the target height and width with zeros.
        """
        padding = (
            0, max_width - image.size(2),  # pad width (left, right)
            0, max_height - image.size(1)  # pad height (top, bottom)
        )
        return F.pad(image, padding, mode='constant', value=0)

    # Pad all images to the maximum height and width
    img1_batch = torch.stack([pad_image(img, max_height, max_width) for img in img1_list], dim=0)
    img2_batch = torch.stack([pad_image(img, max_height, max_width) for img in img2_list], dim=0)
    img3_batch = torch.stack([pad_image(img, max_height, max_width) for img in img3_list], dim=0)

    return img1_batch, img2_batch, img3_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path',
        type=str,
        help="Path to directory containing training dataset.",
        required=True
    )
    parser.add_argument(
        '--val_path',
        type=str,
        help="Path to directory containing validation dataset.",
        required=True
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        required=True
    )
    parser.add_argument(
        '-b',
        '--backbone',
        type=str,
        help="Network backbone from torchvision.models to be used in the siamese network.",
        default="resnet18"
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        help="Learning Rate",
        default=1e-4
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Number of epochs to train",
        default=1000
    )
    parser.add_argument(
        '--batch',
        type=int,
        help="Batch size",
        default=32
    )
    parser.add_argument(
        '-s',
        '--save_after',
        type=int,
        help="Model checkpoint is saved after each specified number of epochs.",
        default=25
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset   = Dataset(args.train_path, shuffle_triplets=True, augment=True)
    val_dataset     = Dataset(args.val_path, shuffle_triplets=False, augment=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, drop_last=True, collate_fn=custom_collate)
    val_dataloader   = DataLoader(val_dataset, batch_size=args.batch, collate_fn=custom_collate)

    model = TripletNetwork(backbone=args.backbone)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)  # TripletMarginLoss with default margin and p

    writer = SummaryWriter(os.path.join(args.out_path, "summary"))

    best_val = float('inf')

    for epoch in range(args.epochs):
        print("[{} / {}]".format(epoch, args.epochs))
        model.train()

        train_losses = []
        correct = 0
        total = 0

        # Training Loop Start
        for (img1, img2, img3) in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training"):
            img1, img2, img3 = map(lambda x: x.to(device), [img1, img2, img3])

            anchor = model(img1)
            positive = model(img2)
            negative = model(img3)
            loss = criterion(anchor, positive, negative)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            pos_similarity = cosine_similarity(anchor, positive)
            neg_similarity = cosine_similarity(anchor, negative)
            correct += (pos_similarity > neg_similarity).sum().item()
            total += img1.size(0)

            accuracy = correct / total if total > 0 else 0

        writer.add_scalar('train_loss', sum(train_losses)/len(train_losses), epoch)
        writer.add_scalar('train_acc', accuracy, epoch)

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(train_losses)/len(train_losses), accuracy))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        val_losses = []
        correct = 0
        total = 0

        for (img1, img2, img3) in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} Validation"):
            img1, img2, img3 = map(lambda x: x.to(device), [img1, img2, img3])

            with torch.no_grad():
                anchor = model(img1)
                positive = model(img2)
                negative = model(img3)
                loss = criterion(anchor, positive, negative)

            val_losses.append(loss.item())

            pos_similarity = cosine_similarity(anchor, positive)
            neg_similarity = cosine_similarity(anchor, negative)

            correct += (pos_similarity > neg_similarity).sum().item()
            total += img1.size(0)

        val_loss = sum(val_losses)/max(1, len(val_losses))
        writer.add_scalar('val_loss', val_loss, epoch)
 
        accuracy = correct / total if total > 0 else 0
        writer.add_scalar('val_acc', accuracy, epoch)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, accuracy))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "best.pth")
            )            

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "epoch_{}.pth".format(epoch + 1))
            )
