import os
import shutil
import random

# Define the path to the raw images folder
raw_dir = 'raw'

# Define the paths to the destination train and test folders
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each person's folder inside the 'raw' directory
for person in os.listdir(raw_dir):
    person_path = os.path.join(raw_dir, person)

    # Skip if it's not a directory (e.g., if there's a file)
    if not os.path.isdir(person_path):
        continue

    # List all image files for this person
    images = os.listdir(person_path)
    images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Check if the person has at least 40 images
    if len(images) < 40:
        print(f"⚠️ Skipping {person} - not enough images ({len(images)} found)")
        continue

    # Shuffle the image list randomly
    random.shuffle(images)

    # Take the first 30 images for training and next 10 for testing
    train_images = images[:30]
    test_images = images[30:40]

    # Create the person's folder in the train and test directories
    os.makedirs(os.path.join(train_dir, person), exist_ok=True)
    os.makedirs(os.path.join(test_dir, person), exist_ok=True)

    # Copy training images
    for img in train_images:
        src = os.path.join(person_path, img)
        dst = os.path.join(train_dir, person, img)
        shutil.copyfile(src, dst)

    # Copy testing images
    for img in test_images:
        src = os.path.join(person_path, img)
        dst = os.path.join(test_dir, person, img)
        shutil.copyfile(src, dst)

    # Print success message
    print(" 30 images copied to train, 10 to test",person)
