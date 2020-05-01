# Face Clustering

Using [facenet](https://github.com/davidsandberg/facenet) for generating face representations, this package performs face clustering on faces found on videos and/or images.

Here is the overall pipeline:
![alt text](https://github.com/iamsoroush/facekoo/blob/master/Untitled%20Diagram.jpg "Overall pipeline")


# Results

Let's find all individual persons(faces) who appears in the following video:

[![Samplr video](http://img.youtube.com/vi/y3d9mBBApQA/0.jpg)](http://www.youtube.com/watch?v=y3d9mBBApQA)


You can see the found clusters in a 2-dimensional representation of embedding space, which is generated using TSNE:

![alt text](https://github.com/iamsoroush/facekoo/blob/master/tsne.png "Found clusters")

And here are the found clusters(persons):

![alt text](https://github.com/iamsoroush/facekoo/blob/master/clusters.png "Persons")


# Usage

1. Install requirements:
..*`python==3.7`
..*`tensorflow-gpu==1.13.1`
..*`keras==2.3.1`
..*`dlib==19.19.0`
..*`jupyter`
..*`opencv==3.4.2`
..*`scikit-learn==0.22.1`
..*`tqdm`

2. Download [facenet model](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) and place it in _models_ directory.
3. Download dlib's [cnn-face-detector](http://dlib.net/files/mmod_human_face_detector.dat.bz2) model and unzip it in _models_ directory.
4. Download dlib's [shape-predictor](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2) model and unzip it in _models_ directory.

For finding the individual persons in a folder of images, inspect this notebook, and for processing a video take a look at this one.
