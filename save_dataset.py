from Extract_Face import load_dataset
from numpy import savez_compressed

# Load the dataset (train and test)
trainX, trainy = load_dataset('dataset/train/')
testX, testy = load_dataset('dataset/test/')

# Save into compressed npz file
savez_compressed('face-dataset.npz', trainX=trainX, trainy=trainy, testX=testX, testy=testy)

print("[INFO] Dataset saved successfully as face-dataset.npz")
