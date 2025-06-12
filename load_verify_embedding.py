import numpy as np

# Load the compressed .npz file containing face embeddings and labels
data = np.load('face-embeddings.npz')

# Print the keys (arrays) stored in the file
print("[INFO] Keys in the file:", data.files)

# Extract training and testing data (embeddings and labels)
trainX, trainy = data['trainX'], data['trainy']
testX, testy = data['testX'], data['testy']

# Display the shapes (dimensions) of the loaded arrays
print("[INFO] Train embeddings shape: ",trainX.shape)  # (number of training samples)
print("[INFO] Train labels shape: ",trainy.shape)      # (number of training labels)
print("[INFO] Test embeddings shape: ",testX.shape)    # (number of test samples)
print("[INFO] Test labels shape: ",testy.shape)        # (number of test labels)

# Print a few sample labels from the training and test sets
print("[INFO] Sample train labels: ",trainy[:5])       # First 5 labels from training set
print("[INFO] Sample test labels: ",testy[:5])         # First 5 labels from test set

# Show the first embedding vector (just the first 10 values for readability)
print("[INFO] First train embedding vector (truncated): ",trainX[0][:10])

# Optional: Check how many unique individuals are in the dataset
unique_train_labels = np.unique(trainy)
unique_test_labels = np.unique(testy)
print("[INFO] Number of unique individuals in train set: ",len(unique_train_labels))
print("[INFO] Number of unique individuals in test set: ",len(unique_test_labels))

# Optional: Count how many samples per individual (just for train set)
from collections import Counter
train_counts = Counter(trainy)
print("[INFO] Number of samples per person in train set:")
for person, count in train_counts.items():
    print("  - {}: {} sample".format(person, count))
