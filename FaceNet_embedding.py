from keras_facenet import FaceNet # FaceNet pre-trained model
import numpy as np

# Load the pre-trained FaceNet model
embedder = FaceNet()

# Function to compute the face embedding from a face image
def get_embedding(model, face_pixels):
    """
    Generates an embedding vector for a given face image using a pre-trained model.

    Parameters:
    - model: The pre-trained embedding model (e.g., FaceNet).
    - face_pixels: A face image of shape (160, 160, 3) as a NumPy array.

    Returns:
    - A 128-dimensional embedding vector that represents the face.
    """

    # Ensure the input face image is in float32 format (required for model input)
    face_pixels = face_pixels.astype('float32')

    # Normalize the pixel values by centering them (zero mean) and scaling (unit variance)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # Add an extra dimension so the input becomes a batch of one image
    # Model expects input shape: (batch_size, 160, 160, 3)
    samples = np.expand_dims(face_pixels, axis=0)

    # Generate the embedding for the face
    # Output is a 128-dimensional vector (1x128)
    yhat = model.embeddings(samples)

    # Return the embedding vector (flattened)
    return yhat[0]
