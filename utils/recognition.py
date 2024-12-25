from keras_facenet import FaceNet
import numpy as np
import pickle
import os

class FaceRecognizer:
    def __init__(self):
        self.embedder = FaceNet()
        self.known_embeddings = self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists('embeddings/embeddings.pkl'):
            with open('embeddings/embeddings.pkl', 'rb') as f:
                return pickle.load(f)
        return {}

    def save_embeddings(self):
        with open('embeddings/embeddings.pkl', 'wb') as f:
            pickle.dump(self.known_embeddings, f)

    def add_face(self, name, face):
        embedding = self.embedder.embeddings(np.array([face]))[0]
        self.known_embeddings[name] = embedding
        self.save_embeddings()

    def recognize(self, face):
        embedding = self.embedder.embeddings(np.array([face]))[0]
        min_distance = float('inf')
        best_match = "Unknown"
        for name, known_embedding in self.known_embeddings.items():
            distance = np.linalg.norm(embedding - known_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = name
        return best_match