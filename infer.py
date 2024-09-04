import cv2
import argparse
import torch
from siamese import SiameseNetwork
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # изменение размера изображения на 224x224
        transforms.ToTensor(),  # преобразование изображения в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # нормализация
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # добавление измерения batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image1',
        type=str,
        help="Path to first image of the pair.",
        required=True
    )
    parser.add_argument(
        '--image2',
        type=str,
        help="Path to second image of the pair.",
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
    model = SiameseNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image1 = load_image(args.image1)
    image2 = load_image(args.image2)
            
    feat1 = model.backbone(image1)
    feat2 = model.backbone(image2)
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)
    cosine_sim = F.cosine_similarity(feat1, feat2, dim=1)
    euclidean_distance = torch.norm(feat1 - feat2, p=2, dim=1)
    e = 1/(1+euclidean_distance)
    print(F"Cosine = {round(cosine_sim.item(), 2)}")
    print(F"1/Distance = {round(e.item(), 2)}")

    similarity = model(image1, image2)
    print(F"Similarity = {round(similarity.item(), 2)}")
