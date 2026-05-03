"""
Module 6 Week B — Core Skills Drill: Embedding Basics

Complete the three functions below to load, query, and compare
pre-trained GloVe word embeddings.
"""

import numpy as np


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array of shape (50,).
    """
    embeddings = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings



def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors.

    Returns a float in [-1, 1]. If either vector has zero norm, return 0.0.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def nearest_neighbors(word, embeddings, n=5):
    """Find the n most similar words to the given word.

    Returns a list of (word, score) tuples sorted by similarity descending,
    excluding the query word itself.
    """
    if word not in embeddings:
        return []
    word_vector = embeddings[word]
    similarities = []
    for w, v in embeddings.items():
        if w != word:
            sim = cosine_similarity(word_vector, v)
            similarities.append((w, sim))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]


if __name__ == "__main__":
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} word vectors")

        # Task 2: Word similarity
        sim = cosine_similarity(glove.get("king", np.zeros(50)),
                                glove.get("queen", np.zeros(50)))
        if sim is not None:
            print(f"cosine('king', 'queen') = {sim:.4f}")

        sim2 = cosine_similarity(glove.get("king", np.zeros(50)),
                                 glove.get("banana", np.zeros(50)))
        if sim2 is not None:
            print(f"cosine('king', 'banana') = {sim2:.4f}")

        # Task 3: Nearest neighbors
        neighbors = nearest_neighbors("king", glove, n=5)
        if neighbors:
            print(f"Nearest to 'king': {neighbors}")
