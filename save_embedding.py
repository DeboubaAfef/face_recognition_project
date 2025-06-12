from numpy import load, savez_compressed
from FaceNet_embedding import get_embedding, embedder
import numpy as np

# Load the saved dataset with face images and labels
data = load('C:face-dataset.npz')
trainX, trainy = data['trainX'], data['trainy']
testX, testy = data['testX'], data['testy']

# Prepare arrays to store embeddings
# Create empty arrays to hold embeddings for train and test sets
train_embeddings = []
test_embeddings = []

# Generate embeddings for training faces
for face_pixels in trainX:
    embedding_vector = get_embedding(embedder, face_pixels)
    train_embeddings.append(embedding_vector)
train_embeddings = np.asarray(train_embeddings)

# Generate embeddings for testing faces
for face_pixels in testX:
    embedding_vector = get_embedding(embedder, face_pixels)
    test_embeddings.append(embedding_vector)
test_embeddings = np.asarray(test_embeddings)

# Save the embeddings and labels into a compressed .npz file

savez_compressed('face-embeddings.npz',
                 trainX=train_embeddings, trainy=trainy,
                 testX=test_embeddings, testy=testy)

print("[INFO] Face embeddings generated and saved successfully in 'face-embeddings.npz'")
