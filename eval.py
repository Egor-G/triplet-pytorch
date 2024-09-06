import os
import argparse

import cv2
import numpy as np

import torch
from torchvision import transforms
from torch.nn.functional import cosine_similarity

from triplet import TripletNetwork
from PIL import Image


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
            # Берем имя файла без расширения, чтобы оно совпадало с подпапками
            embeddings[os.path.splitext(file_name)[0]] = torch.from_numpy(np.load(file_path))
    return embeddings


def find_most_similar_image(model, input_image_path, embeddings, device):
    input_tensor = load_image(input_image_path).to(device)
    with torch.no_grad():
        input_embedding = model(input_tensor).cpu().squeeze(0)  # Убираем размерность batch

    # Compute cosine similarity
    max_similarity = -1
    best_match = None
    for class_name, stored_embedding in embeddings.items():
        # Ensure the stored embedding is on the same device as input embedding
        stored_embedding = stored_embedding.to(device)
        similarity = cosine_similarity(input_embedding.unsqueeze(0), stored_embedding.unsqueeze(0)).item()
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = class_name

    return best_match, max_similarity


def find_top_k_similar_images(model, input_image_path, embeddings, device, k=5):
    input_tensor = load_image(input_image_path).to(device)
    with torch.no_grad():
        input_embedding = model(input_tensor).cpu().squeeze(0)

    similarities = []
    for class_name, stored_embedding in embeddings.items():
        stored_embedding = stored_embedding.to(device)
        similarity = cosine_similarity(input_embedding.unsqueeze(0), stored_embedding.unsqueeze(0)).item()
        similarities.append((class_name, similarity))

    # Сортируем по убыванию сходства и возвращаем топ-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def process_validation_set(model, val_path, embeddings, device):
    correct = 0
    total = 0

    # Проход по всем подпапкам в val_path
    for root, dirs, files in os.walk(val_path):
        for file_name in files:
            if file_name.endswith(('.jpg', '.png')):
                total += 1
                file_path = os.path.join(root, file_name)
                true_class = os.path.basename(os.path.normpath(root))  # Имя подпапки как истинный класс
                
                # Находим наиболее вероятное сходство с загруженными эмбеддингами
                predicted_class, similarity = find_most_similar_image(model, file_path, embeddings, device)
                
                #similarities = find_top_k_similar_images(model, file_path, embeddings, device, k=5)
                #print(f"\nTrue: {true_class}")
                #for i, (pred, sim) in enumerate(similarities):
                #    print(f"Top {i+1}: {pred}:{sim:.4f}")
                
                # Сравниваем предсказанный класс с истинным
                if predicted_class == true_class:
                    correct += 1

                accuracy = correct / total if total > 0 else 0
            print(f"Accuracy: {accuracy:.2f} ({correct}/{total})")

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

    args = parser.parse_args()

    # Установка устройства на CUDA, если доступно, иначе CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загрузка модели
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TripletNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Загрузка эмбеддингов
    embeddings = load_embeddings(args.embeddings)
    
    # Обработка валидационного набора
    process_validation_set(model, args.val_path, embeddings, device)
