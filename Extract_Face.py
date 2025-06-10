import cv2 # OpenCv library for image processing and face detection

# Load HAAR cascade model once (used to detect faces in images)
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Function to extract a single face from an image
def extract_face(filename, required_size=(160, 160)):
    """
    Detects a face in the image and returns the cropped face resized to (160x160).

    Parameters:
    - filename (str): path to the image file.
    - required_size (tuple): size to resize the face (160, 160) for FaceNet.

    Returns:
    - face_array (numpy array): the cropped and resized face or None if not found.
    """
    image = cv2.imread(filename)  # Read the image from disk
    if image is None:
        # if the image couldn't be read , print an error and return None
        print("[ERROR] Could not read image: {}".format(filename))
        return None

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If no faces found
    if len(faces) == 0:
        return None

    # Assume the first detected face
    x, y, w, h = faces[0] # x, y, w, h are coordinates and size of the rectangle
    face = image[y:y + h, x:x + w]

    # Resize to required size
    face_resized = cv2.resize(face, required_size)

    return face_resized # Return the final image

# Load multiple faces from a single directory

import os # Module to interact with the operating system
from numpy import asarray # To convert lists to NumPy arrays

def load_faces(directory):
    """
    Loads all faces from a directory of images using extract_face.

    Parameters:
    - directory (str): Path to the directory containing face images.

    Returns:
    - faces (list of numpy arrays): List of extracted face images.
    """
    faces = []  # List to store the face arrays
    for filename in os.listdir(directory):  # Loop through each file in the directory
        path = os.path.join(directory, filename)  # Get full path of the image file
        face = extract_face(path)  # Call extract_face on the image
        if face is not None:
            faces.append(face)  # Add the extracted face to the list
    return faces  # Return the list of faces

# Load a dataset organized in folders per person (train or test)

def load_dataset(directory):
    """
    Loads a dataset from a parent directory where each subdirectory is a class label.

    Parameters:
    - directory (str): Parent directory containing subdirectories for each person.

    Returns:
    - X (list): List of face images (NumPy arrays).
    - y (list): List of corresponding labels.
    """
    X, y = [], []  # X = face images, y = labels
    for subdir in os.listdir(directory):  # Loop through each subfolder (person)
        subpath = os.path.join(directory, subdir)  # Full path to the person's folder
        if not os.path.isdir(subpath):
            continue  # Skip if it's not a directory (just a file)
        faces = load_faces(subpath)  # Load faces from this folder
        labels = [subdir] * len(faces)  # Create a list of labels (same name) for each face
        X.extend(faces)  # Add faces to main list
        y.extend(labels)  # Add corresponding labels
    return asarray(X), asarray(y)  # Return as NumPy arrays

