import numpy as np

# Load the .npz file
data = np.load('face-dataset.npz')

# See the keys (array names) stored inside
print(data.files)  #  ['trainX', 'trainy', 'testX', 'testy']

# Access arrays by their keys
trainX = data['trainX']
trainy = data['trainy']
testy = data['testy']
print("Train images shape:", trainX.shape)
print("Train labels shape:", trainy.shape)
print("All training labels:")
# Print all unique labels in training set
print("Unique labels in training set:", np.unique(trainy))

# Print all unique labels in test set
print("Unique labels in test set:", np.unique(testy))