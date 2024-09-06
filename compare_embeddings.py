import os
import random
import numpy as np
import torch
from torch.nn.functional import cosine_similarity

def load_embeddings(directory):
    """Загружает все файлы .npy с эмбеддингами из указанной директории"""
    embeddings = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy'):
            file_path = os.path.join(directory, file_name)
            embeddings[file_name] = torch.from_numpy(np.load(file_path))
    return embeddings

def compare_random_embeddings(embeddings, num_pairs=5, device='cpu'):
    """Сравнивает случайные пары эмбеддингов и выводит косинусное сходство"""
    embedding_names = list(embeddings.keys())

    if len(embedding_names) < 2:
        print("Недостаточно эмбеддингов для сравнения.")
        return

    # Выбираем случайные пары
    for _ in range(num_pairs):
        emb1_name, emb2_name = random.sample(embedding_names, 2)

        emb1 = embeddings[emb1_name].to(device)
        emb2 = embeddings[emb2_name].to(device)

        # Вычисляем косинусное сходство
        similarity = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        print(f"Cosine similarity between {emb1_name} and {emb2_name}: {similarity:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e',
        '--embeddings',
        type=str,
        help="Path to directory containing .npy embedding files.",
        required=True
    )
    parser.add_argument(
        '-n',
        '--num_pairs',
        type=int,
        help="Number of random pairs to compare.",
        default=5
    )

    args = parser.parse_args()

    # Загружаем эмбеддинги
    embeddings = load_embeddings(args.embeddings)

    # Сравниваем случайные пары эмбеддингов
    compare_random_embeddings(embeddings, num_pairs=args.num_pairs)
