This part is about preparing the data for face recognition. Here’s what we did step by step:

1. Data Collection

We collected many pictures of different people (5 famous person) automatically using a tool (like GoogleImageCrawler).
These pictures are saved in folders named after each person.

2. Data Splitting

We separated the pictures into two groups:

Training set (used to teach the model)
Testing set (used to check how good the model is)

Each group is saved in its own folder:→ dataset/train/ and dataset/test/.

3. Face Extraction

We used a face detection tool called HAAR Cascade from OpenCV. It detects and crops faces from each image.

Each face is resized to 160x160 pixels (required input size for the FaceNet model).

4. Dataset Loading

We created custom functions that:

Go through all folders and images
Extract faces from each image
Assign the correct label (person’s name, based on the folder)

This results in:

A list of face images (as NumPy arrays)
A list of corresponding labels (strings)

5. Saving the Dataset

We saved the face arrays and labels into one compressed file: face-dataset.npz (using numpy.savez_compressed)
This makes it easier to reuse later without repeating the extraction steps.

⚠️ Issue: Incomplete Dataset (145 instead of 150)

While verifying the dataset using load_verify.py, we noticed a problem:

Only 145 face images were loaded, not the expected 150 (5 persons × 30 training images).

This meant that 5 images had no detectable face, and were skipped during extraction.

🛠️ Solution: Detection Testing with test.py

To fix this, we wrote a small script (test.py) to check which training images failed.

✅ Load and Verify

The next step is to load the .npz file and verify the data before training FaceNet. This includes:

Printing sample faces with their labels
Checking the shape of the arrays
Making sure all classes are correctly represented

