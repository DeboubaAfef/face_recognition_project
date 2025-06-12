Phase 1:
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
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Phase 2: 
Generating Face EmbeddingsObjective 

Steps Taken:

Installing Required Libraries

We started by installing all necessary libraries including:

keras-facenet

tensorflow, numpy

Using FaceNet for Embedding Extraction

We implemented FaceNet_embedding.py, which:

Loads the pre-trained FaceNet model.
Converts face images into 512-dimensional embedding vectors using the get_embedding() function.

We tested another option using VGG16_embedding.py, but did not use it in the final pipeline.

Why FaceNet Instead of VGG16?

FaceNet is specifically trained for face recognition and person identification.
VGG16 is a general-purpose image classification model (better for classes like humans, dogs, cats...).

Therefore, FaceNet was a better choice for generating robust face embeddings.

Saving the Embeddings

We used save_embedding.py to:

Load the original dataset of aligned face images (face-dataset.npz).
Generate embeddings for each face image using FaceNet.
Save the resulting embeddings and their labels into a compressed file: face-embeddings.npz.

 Challenges Faced:

At first, there was some confusion about the embedding size:

We expected a 128-dimensional vector (as in some FaceNet implementations), but our model returned 512-dimensional vectors.

After verifying that it’s still consistent with the keras-facenet version, we decided to keep the 512-d embedding because:

It contains more information.
It resulted in no errors.
It may improve classification accuracy.

Output:

File: face-embeddings.npz

Contains:

trainX: 512-d embeddings of training face images.
trainy: corresponding labels.
testX: 512-d embeddings of testing face images.
testy: corresponding labels.

Conclusion:

We successfully completed Phase 2 by generating meaningful embeddings using FaceNet. The dataset is now ready for training a classifier.
