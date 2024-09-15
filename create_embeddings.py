import cv2
import argparse
import torch
import os
import glob
from triplet import TripletNetwork
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import random

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 320)),
        transforms.ToTensor(),  # преобразование изображения в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # нормализация
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # добавление измерения batch

def extract_embeddings(model, image_paths, device):
    embeddings = []
    for image_path in image_paths:
        image_tensor = load_image(image_path)
        with torch.no_grad():
            make_pred, embedding = model(image_tensor)  # извлекаем make и embedding
            embedding = embedding.cpu().numpy()  # преобразуем embedding в numpy
        embeddings.append(embedding)
    return np.vstack(embeddings)

def find_image_directories(directory):
    # Список директорий, содержащих изображения
    image_dirs = []

    # Проходим по всем вложенным директориям
    for root, dirs, files in os.walk(directory):
        # Ищем изображения в текущей директории
        image_files = glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png"))

        # Если нашли хотя бы одно изображение, добавляем текущую директорию в список
        if image_files:
            image_dirs.append(root)
    
    return image_dirs

def find_images_in_directory(directory):
    # Используем glob для поиска изображений с расширениями .jpg и .png
    image_paths = glob.glob(os.path.join(directory, "**/*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(directory, "**/*.png"), recursive=True)
    
    # Если файлов меньше 50, возвращаем все
    if len(image_paths) <= 50:
        return image_paths
    
    # Если файлов больше 50, выбираем 50 случайных уникальных
    return random.sample(image_paths, 50)

def format_directory_path(directory_path):
    # Разбиваем путь на составляющие и соединяем их с помощью подчеркиваний
    formatted_path = "_".join(directory_path.split(os.sep))
    return formatted_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help="Path to output directory.",
        required=True
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        required=True
    )
    parser.add_argument(
        '-n',
        '--classes',
        type=int,
        help="Number of classes",
    )
    parser.add_argument(
        'input_dirs',
        nargs='+',
        help="Path(s) to input directory/directories."
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TripletNetwork(backbone=checkpoint['backbone'], num_classes=args.classes)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Process each input directory
    for input_dir in args.input_dirs:
        image_directories = find_image_directories(input_dir)
        for img_dir in image_directories:
            image_paths = find_images_in_directory(img_dir)
            if not image_paths:
                print(f"No images found in {img_dir}")
                continue
        
            # Extract embeddings
            embeddings = extract_embeddings(model, image_paths, device)
            mean_embedding = np.mean(embeddings, axis=0)
        
            # Save the mean embedding to a file
            output_file = os.path.join(args.output, f"{os.path.basename(format_directory_path(img_dir))}.npy")
            np.save(output_file, mean_embedding)
            print(f"Saved embeddings for {img_dir} to {output_file}")
