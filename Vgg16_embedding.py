from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import cv2

# Load the VGG16 model without the top (classification) layers
# This gives us only the feature extractor part of the model
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

# Flatten the output to get a 1D feature vector
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to compute the embedding from a face image
def get_vgg16_embedding(face_pixels):
    """
    Generates a VGG16-based feature embedding for a given face image.

    Parameters:
    - face_pixels: A face image (NumPy array) of shape (224, 224, 3)

    Returns:
    - A flattened vector (features extracted from the VGG16 model)
    """

    # Convert image to float32 and preprocess it for VGG16
    face_pixels = face_pixels.astype('float32')
    face_pixels = preprocess_input(face_pixels)

    # Add batch dimension: model expects (batch_size, 224, 224, 3)
    sample = np.expand_dims(face_pixels, axis=0)

    # Pass through the model to get the feature map
    feature_map = model.predict(sample)

    # Flatten the feature map to get a 1D vector
    embedding = feature_map.flatten()

    return embedding

# Example usage
if __name__ == "__main__":
    # Load an example face image
    image_path = "dataset/train/Angelina_Jolie/000001.jpg"  # Replace with an image from dataset
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read image.")
    else:
        # Resize image to 224x224 for VGG16
        image = cv2.resize(image, (224, 224))

        # Get the embedding
        embedding = get_vgg16_embedding(image)

        # Print the shape of the embedding
        print("Embedding shape:", embedding.shape)  # (7*7*512,) = (25088,)
