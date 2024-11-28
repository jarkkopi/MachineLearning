import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

vocabulary_file='word_embeddings.txt'

def calculate_distances(word_vec, W):
    distances = np.linalg.norm(W - word_vec, axis=1)
    return distances

def find_nearest_neighbors(word_vec, W, k=3):
    distances = calculate_distances(word_vec, W)

    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices, distances[nearest_indices]

def word_search(W,word):
    word_idx = vocab[word]
    word_vec = W[word_idx].reshape(1, -1)
    indices, distances = find_nearest_neighbors(word_vec, W, k=3)
    print(f"Nearest words to {word}:")
    for i in range(len(indices)):
        print(f"{ivocab[indices[i]]} (distance: {distances[i]})")

print('Read words...')
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# Main loop for analogy
answer = input("Word search (w) or search analogy (s): ")
if answer == "w":
    while (word!="EXIT"):
        word = input("Enter word to search: ")
        word_search(W,word)
    exit
else:
    while True:
        input_term = input("\nEnter word (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            input_term = input_term.split("-")
            x = input_term[0]
            y = input_term[1]
            z = input_term[2]
            vecx = W[vocab[x]]
            vecy = W[vocab[y]]
            vecz = W[vocab[z]]
            analogy_vector = vecz + (vecy - vecx)
    
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(W)
            distances, indices = nbrs.kneighbors(analogy_vector.reshape(1, -1))
            result = [ivocab[indices[0][i]] for i in range(0,10) if ivocab[indices[0][i]] not in input_term]
            print(f"{x} is to {y} as {z} is to {result[:2]}")

