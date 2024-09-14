import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from triplet import TripletNetwork
from libs.dataset import Dataset

def accuracy(preds, targets):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, required=True, help="Path to directory containing training dataset.")
    parser.add_argument('--val_path', type=str, required=True, help="Path to directory containing validation dataset.")
    parser.add_argument('-o', '--out_path', type=str, required=True, help="Path for outputting model weights and tensorboard summary.")
    parser.add_argument('-b', '--backbone', type=str, default="resnet18", help="Network backbone from torchvision.models to be used.")
    parser.add_argument('-r', '--resume', type=str, default="", help="Resume train, path to saved model.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('-e', '--epochs', type=int, default=1000, help="Number of epochs to train")
    parser.add_argument('--batch', type=int, default=32, help="Batch size")
    parser.add_argument('-s', '--save_after', type=int, default=25, help="Model checkpoint is saved after each specified number of epochs.")

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Dataset(args.train_path, shuffle_triplets=True, augment=True)
    val_dataset = Dataset(args.val_path, shuffle_triplets=False, augment=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch)

    if args.resume != "":
        model = TripletNetwork(backbone=args.backbone, num_classes=len(train_dataset.class_names))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        start_epoch = 0
    else:
        checkpoint = torch.load(args.resume, map_location=device)
        model = TripletNetwork(backbone=checkpoint['backbone'], num_classes=len(train_dataset.class_names))
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    triplet_criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    class_criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(os.path.join(args.out_path, "summary"))
    best_val = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"[{epoch + 1} / {args.epochs}]")
        model.train()

        train_losses = []
        class_accuracies = []
        correct = 0
        total = 0

        for (img1, img2, img3, target_make) in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training"):
            img1, img2, img3, target_make = map(lambda x: x.to(device), [img1, img2, img3, target_make])

            # Forward pass
            make_pred1, anchor_embedding = model(img1)
            make_pred2, positive_embedding = model(img2)
            make_pred3, negative_embedding = model(img3)

            # Calculate losses
            triplet_loss = triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
            make_loss = (class_criterion(make_pred1, target_make) +
                         class_criterion(make_pred2, target_make) +
                         class_criterion(make_pred3, target_make)) / 3.0
            loss = triplet_loss + 2 * make_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            class_accuracies.append(accuracy(make_pred1, target_make))

            pos_similarity = F.cosine_similarity(anchor_embedding, positive_embedding)
            neg_similarity = F.cosine_similarity(anchor_embedding, negative_embedding)
            correct += (pos_similarity > neg_similarity).sum().item()
            total += img1.size(0)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_class_accuracy = sum(class_accuracies) / len(class_accuracies)
        avg_embeddings_accuracy = correct / total if total > 0 else 0

        writer.add_scalar('train_loss', avg_train_loss, epoch)
        writer.add_scalar('train_class_accuracy', avg_class_accuracy, epoch)
        writer.add_scalar('train_embeddings_accuracy', avg_embeddings_accuracy, epoch)

        print(f"\tTraining: Loss={avg_train_loss:.2f}\t Enbeddings accuracy={avg_embeddings_accuracy:.2f}\t Classify accuracy={avg_class_accuracy:.2f}")

        # Evaluation Loop Start
        model.eval()

        val_losses = []
        val_class_accuracies = []
        correct = 0
        total = 0

        for (img1, img2, img3, target_make) in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} Validation"):
            img1, img2, img3, target_make = map(lambda x: x.to(device), [img1, img2, img3, target_make])

            with torch.no_grad():
                make_pred1, anchor_embedding = model(img1)
                make_pred2, positive_embedding = model(img2)
                make_pred3, negative_embedding = model(img3)
                triplet_loss = triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
                make_loss = (class_criterion(make_pred1, target_make) +
                             class_criterion(make_pred2, target_make) +
                             class_criterion(make_pred3, target_make)) / 3.0
                loss = triplet_loss + make_loss

            val_losses.append(loss.item())
            val_class_accuracies.append(accuracy(make_pred1, target_make))

            pos_similarity = F.cosine_similarity(anchor_embedding, positive_embedding)
            neg_similarity = F.cosine_similarity(anchor_embedding, negative_embedding)

            correct += (pos_similarity > neg_similarity).sum().item()
            total += img1.size(0)

        avg_val_loss = sum(val_losses) / max(1, len(val_losses))
        avg_val_class_accuracy = sum(val_class_accuracies) / len(val_class_accuracies)
        avg_val_embeddings_accuracy = correct / total if total > 0 else 0

        writer.add_scalar('val_loss', avg_val_loss, epoch)
        writer.add_scalar('val_class_accuracy', avg_val_class_accuracy, epoch)
        writer.add_scalar('val_embeddings_accuracy', avg_val_embeddings_accuracy, epoch)

        print(f"\tValidation: Loss={avg_val_loss:.2f}\t Embeddings accuracy={avg_val_embeddings_accuracy:.2f}\t Classify accuracy={avg_val_class_accuracy:.2f}")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "class_names": train_dataset.class_names,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "best.pth")
            )            

        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "class_names": train_dataset.class_names,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, f"epoch_{epoch + 1}.pth")
            )
