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
        transforms.Resize(224),
        transforms.CenterCrop((224, 320)),
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

def find_most_similar_image(input_embedding, embeddings):
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
        '-n',
        '--classes',
        type=int,
        help="Number of classes",
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help="Path(s) to input image file(s)."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TripletNetwork(backbone=checkpoint['backbone'], num_classes=args.classes)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint['class_names']
    model.eval()

    for input_file in args.input_files:
        if not os.path.isfile(input_file):
            print(f"{input_file} is not a valid file.")
            continue
        
        input_tensor = load_image(input_file).to(device)
        with torch.no_grad():
            input_class, input_embedding = model(input_tensor)
        input_embedding = input_embedding.cpu().squeeze(0)
        predicted_class_idx = torch.argmax(input_class, dim=1).item()
        predicted_class_name = class_names[predicted_class_idx]

        embeddings = load_embeddings(args.embeddings + "/" + predicted_class_name)
        best_match, max_similarity = find_most_similar_image(input_embedding, embeddings)
        if best_match:
            basename = os.path.basename(input_file)
            print(f"{basename}: Make: {predicted_class_name}, Model: {best_match.rsplit('.', 1)[0]}, {max_similarity:.4f}")
        else:
            print(f"No matching embeddings found")
