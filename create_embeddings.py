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

def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),  # преобразование изображения в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # нормализация
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # добавление измерения batch

def extract_embeddings(model, image_paths, device):
    embeddings = []
    for image_path in image_paths:
        image_tensor = load_image(image_path).to(device)
        with torch.no_grad():
            embedding = model(image_tensor).cpu().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

def filter_outliers(embeddings, threshold=0.8):
    # Вычисляем начальный средний эмбеддинг
    mean_embedding = np.mean(embeddings, axis=0)
    
    # Вычисляем косинусное сходство каждого эмбеддинга с начальным средним эмбеддингом
    similarities = [F.cosine_similarity(torch.tensor(embedding), torch.tensor(mean_embedding), dim=0).item() for embedding in embeddings]
    
    # Фильтруем эмбеддинги, у которых сходство выше порогового значения
    filtered_embeddings = [embeddings[i] for i in range(len(embeddings)) if similarities[i] >= threshold]
    
    return np.mean(filtered_embeddings, axis=0), filtered_embeddings

def find_images_in_directory(directory):
    # Используем glob для поиска изображений с расширениями .jpg и .png
    image_paths = glob.glob(os.path.join(directory, "**/*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(directory, "**/*.png"), recursive=True)
    return image_paths

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
        '-t',
        '--filter_threshold',
        type=float,
        help="Mean filter out threshold",
        default=0.5
    )
    parser.add_argument(
        'input_dirs',
        nargs='+',
        help="Path(s) to input directory/directories."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TripletNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Process each input directory
    for input_dir in args.input_dirs:
        image_paths = find_images_in_directory(input_dir)
        if not image_paths:
            print(f"No images found in {input_dir}")
            continue
        
        # Extract embeddings
        embeddings = extract_embeddings(model, image_paths, device)
        
        # Filter outliers and calculate the new mean embedding
        mean_embedding, filtered_embeddings = filter_outliers(embeddings, threshold=args.filter_threshold)
        
        # Save the mean embedding to a file
        output_file = os.path.join(args.output, f"{os.path.basename(os.path.normpath(input_dir))}.npy")
        np.save(output_file, mean_embedding)
        print(f"Saved embeddings for {input_dir} to {output_file}")
        print(f"Filtered out {len(embeddings) - len(filtered_embeddings)} outliers")   
