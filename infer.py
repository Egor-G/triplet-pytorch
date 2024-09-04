import os
import argparse
import torch
from triplet import TripletNetwork
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.nn.functional import cosine_similarity
from pathlib import Path

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # изменение размера изображения на 224x224
        transforms.ToTensor(),  # преобразование изображения в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # нормализация
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # добавление измерения batch

def load_embeddings(directory):
    embeddings = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy'):
            file_path = os.path.join(directory, file_name)
            embeddings[file_name] = torch.from_numpy(np.load(file_path))
    return embeddings

def find_most_similar_image(model, input_image_path, embeddings, device):
    input_tensor = load_image(input_image_path).to(device)
    with torch.no_grad():
        input_embedding = model(input_tensor).cpu().squeeze(0)  # Убираем размерность batch

    # Compute cosine similarity
    max_similarity = -1
    best_match = None
    for file_name, stored_embedding in embeddings.items():
        # Ensure the stored embedding is on the same device as input embedding
        stored_embedding = stored_embedding.to(device)
        similarity = cosine_similarity(input_embedding.unsqueeze(0), stored_embedding.unsqueeze(0)).item()
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = file_name

    return best_match, max_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e',
        '--embeddings',
        type=str,
        help="Path to directory containing .npy embedding files.",
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
        'input_files',
        nargs='+',
        help="Path(s) to input image file(s)."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TripletNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    embeddings = load_embeddings(args.embeddings)

    for input_file in args.input_files:
        if not os.path.isfile(input_file):
            print(f"{input_file} is not a valid file.")
            continue

        best_match, max_similarity = find_most_similar_image(model, input_file, embeddings, device)
        if best_match:
            basename = Path(input_file).stem
            print(f"{basename}: {best_match}, {max_similarity:.4f}")
        else:
            print(f"No matching embeddings found")
