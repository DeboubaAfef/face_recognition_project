import os
from Extract_Face import extract_face

directory = 'dataset/train'
for person in os.listdir(directory):
    person_dir = os.path.join(directory, person)
    for filename in os.listdir(person_dir):
        path = os.path.join(person_dir, filename)
        face = extract_face(path)
        if face is None:
            print("[WARNING] No face detected in: ",path)
