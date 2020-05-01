# -*- coding: utf-8 -*-
"""Face clustering module.

   This file contains FaceClustering class.
   It uses facenet's pretrained CNN for feature extraction, and uses predefined
    clustering algorithms for performing clustering on images or video files.

   Instantiate an object from this class, tune the model on your own data if
    needed, and enjoy clustered faces.


   You can use the model in context manager mode, but the session will
    terminate at the end of the 'with' block. When using in standard way,
    the background tensorflow session wouldn't terminate in order to avoid
    redundant operations.
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

import os
import logging

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from .clustering import ROCWClustering
from .face_utils import FaceNet, FaceDetector, FaceAligner


class FaceClusering:

    """Performs clustering on faces, based on embeddings, images or video.

    Naming rules:
        self.name : externally given parameters (as argument), and public
         parameters
        self.name_ : internally created, non-public parameters
        self.name() : public methods
        self._name() : internal methods


    Public methods:
        do_clustering: Clustering on given data samples as array of arrays.
        do_clustering_on_images: Clustering on given list of RGB images.
        do_clustering_on_image_paths: Clustering on images found on specified paths.
        do_clustering_on_video: Clustering on given video path.
        evaluate: Evaluates clustering algorithm on given data.
        evaluate_on_images: Evaluates clustering on given images

    Attributes:
        clusterer_: An instance of ROCWClustering algorithm.
        facenet_model_: An instance of FaceNet model, for generating embeddings.
        face_detector_: An instance of FaceDetector, for detecting faces in images.
        face_aligner_: An instance of FaceAligner, for aligning found faces.

    """

    def __init__(self, config, clustering_alg='dbscan'):

        assert clustering_alg in ('cw', 'dbscan')

        if clustering_alg == 'cw':
            self.clusterer_ = ROCWClustering(k=config.ROCW_K,
                                             metric=config.ROCW_METRIC,
                                             n_iteration=config.ROCW_N_ITERATION,
                                             algorithm=config.ROCW_ALGORITHM)
        else:
            self.clusterer_ = DBSCAN(eps=config.DBSCAN_EPS,
                                     min_samples=config.DBSCAN_MIN_SAMPLES,
                                     metric=config.DBSCAN_METRIC)
        print('clustering algorithm created.')

        self.facenet_model_ = FaceNet(model_path=config.FACENET_MODEL_PATH,
                                      batch_size=config.FACENET_BATCH_SIZE,
                                      image_size=config.FACENET_IMAGE_SIZE,
                                      do_prewhiten=config.FACENET_DO_PREWHITEN,
                                      do_crop=config.FACENET_DO_CROP)
        print('facenet is ready.')

        self.face_detector_ = FaceDetector(cnn_path=config.DLIB_FACE_DETECTOR_PATH)
        print('face detector is ready.')

        self.face_aligner_ = FaceAligner(predictor_path=config.DLIB_FACIAL_SHAPE_PREDICTOR_PATH,
                                         desired_left_eye=config.FACE_ALIGNER_LEFT_EYE,
                                         desired_face_width=config.FACE_ALIGNER_OUTPUT_SIZE)
        print('face aligner is ready.')

        self.valid_images = config.VALID_IMAGES
        self.max_input_height = config.MAX_INPUT_HEIGHT
        self.corr_th_between_frames = config.CORRELATION_THRESHOLD_BETWEEN_FRAMES
        self.vid_sampling_period = config.VIDEO_SAMPLING_PERIOD

    def do_clustering(self, X):

        """Clustering on given samples.

        Args:
            X: an array of arrays, each row is a sample and columns are
             features.

        Returns:
            an array that contains predicted labels for given X.
        """

        labels = self.clusterer_.fit_predict(X)
        return labels

    def do_clustering_on_images(self, images_dir):

        """Clustering on images found on specified folder.

        Finds and aligns faces, saves faces in a new folder called "found_faces" on given path,
         then performs clustering on found faces.

        Args:
            images_dir: folder of images.

        Returns:
            A dictionary that maps each found_face_name to its label.
        """

        faces_dir = os.path.join(images_dir, 'found_faces')
        if not os.path.exists(faces_dir):
            embs, faces = self._detect_align_generate_embs(images_dir)
            if np.any(embs):
                labels = self.clusterer_.fit_predict(embs)
                uniques = np.unique(labels)
                os.mkdir(faces_dir)
                for label in uniques:
                    label_dir = os.path.join(faces_dir, str(label))
                    os.mkdir(label_dir)
                    for ind in np.where(labels == label)[0]:
                        img_path = os.path.join(label_dir, '{}.jpg'.format(ind))
                        self._write_bgr_image(faces[ind], img_path)
                        # skio.imsave(img_path, img_as_ubyte(faces[ind]))
                self.draw_tsne(embs, labels, faces_dir)
                print('results saved on ', faces_dir)
        else:
            print('the found_faces sub-directory already exists at ', images_dir)

    def do_clustering_on_video(self, video_path, num_frames=None):

        """Clustering on given video path.

        Args:
            video_path: complete path to video.
            num_frames: how many frames to read and process from video?

        Returns:
            a dictionary containing cropped faces for each cluster:
                {cluster1: [face1, face2, ...], cluster2: [face1, face2], ...}
        """

        base_dir = os.path.dirname(video_path)
        faces_dir = os.path.join(base_dir, 'found_faces')
        if not os.path.exists(faces_dir):
            embs, faces = self._dag_video(video_path, num_frames)
            if embs:
                labels = self.clusterer_.fit_predict(embs)
                uniques = np.unique(labels)
                os.mkdir(faces_dir)
                for label in uniques:
                    label_dir = os.path.join(faces_dir, str(label))
                    os.mkdir(label_dir)
                    for ind in np.where(labels == label)[0]:
                        img_path = os.path.join(label_dir, '{}.jpg'.format(ind))
                        self._write_bgr_image(faces[ind], img_path)
                        # skio.imsave(img_path, img_as_ubyte(faces[ind]))
                self.draw_tsne(embs, labels, faces_dir)
                print('results saved on ', faces_dir)
        else:
            print('found faces already exists.')

    def show_cluster(self, cluster_dir):

        """Shows all members of given cluster in a single image."""

        face_paths = [os.path.join(cluster_dir, item) for item in os.listdir(cluster_dir) if item.endswith('.jpg')]
        fig = self._plot_faces(face_paths)
        fig.show()

    def _dag_video(self, video_path, num_frames):

        """Detect and align the faces found in given video and generate embeddings for them."""

        vidcap = cv2.VideoCapture(video_path)
        w, h, fps, fc = self._get_vid_specs(vidcap)
        if num_frames is None:
            num_frames = fc

        to_skip = int(self.vid_sampling_period * fps)

        embs = list()
        faces = list()
        last_frame = np.zeros((h, w, 3), dtype=np.int8)
        with tqdm(total=int(num_frames / to_skip)) as pbar:
            for frame_ind in range(num_frames):
                frame = self._read_rgb_frame(vidcap)
                if frame is not None:
                    if frame_ind % to_skip == 0:
                        if np.any(frame):
                            if np.any(last_frame):
                                corr = np.abs(np.corrcoef(last_frame.flatten(), frame.flatten())[0, 1])
                            else:
                                corr = 0
                            if corr < self.corr_th_between_frames:
                                aligned_faces = self._detect_align(frame)
                                if np.any(aligned_faces):
                                    embs.extend(self.facenet_model_.generate_embeddings_from_images(aligned_faces))
                                    faces.extend(aligned_faces)
                                    # for j, face in enumerate(aligned_faces):
                                        # face_path = os.path.join(faces_dir, '{}_{}.jpg'.format(frame_ind, j))
                                        # faces.append(face_path)
                        last_frame = frame.copy()
                        pbar.update(1)
                else:
                    raise Exception('could not read frame {}'.format(frame_ind))

        return embs, faces

    def _detect_align(self, image):

        """Detects faces, aligns and returns them."""

        image_height = image.shape[0]
        # if image_height > self.max_input_height:
        scale_factor = self.max_input_height / image_height
        image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        rects = self.face_detector_.detect_faces(image)
        if not rects:
            return []
        aligned_faces = self.face_aligner_.align(image, rects)
        return aligned_faces

    def _detect_align_generate_embs(self, images_dir):

        """Detects faces, aligns them and returns the generated embeddings for them."""

        image_paths = [os.path.join(images_dir, img_name)
                       for img_name in os.listdir(images_dir) if img_name.lower().endswith(self.valid_images)]

        if not image_paths:
            print('could not find any image in ', images_dir)
            return None, None

        # face_paths = list()
        embs = list()
        faces = list()
        print('processing ...')
        with tqdm(total=len(image_paths)) as pbar:
            for path in image_paths:
                image = self._read_rgb_image(path)
                aligned_faces = self._detect_align(image)
                if aligned_faces:
                    embs.extend(self.facenet_model_.generate_embeddings_from_images(aligned_faces))
                    faces.extend(aligned_faces)
                    # for j, face in enumerate(aligned_faces):
                    #     face_path = os.path.join(face_folder, '{}_{}.jpg'.format(i, j))
                    #     skio.imsave(face_path, face)
                    #     face_paths.append(face_path)
                pbar.update(1)
        # return face_paths, np.array(embs)
        return embs, faces

    def evaluate(self, X, y_true):

        """Evaluates clustering algorithm on given data.

        Args:
            X: an array of arrays
            y_true: 1D array, true labels for each row of X

        Returns:
            pairwise f-measure
        """

        return self.clusterer_.score(X, y_true)

    def evaluate_on_images(self, images, labels):

        """Evaluates clustering on given images.

        Args:
            images: array of rgb images.
            labels: array of labels for given images.

        Returns:
            pairwise f-measure
        """

        embeddings = list()
        flipped_faces = list()
        for image in images:
            rects = self.face_detector_.detect_faces(image)
            if not rects:
                continue
            aligned_faces = self.face_aligner_.align(image, rects)
            for face in aligned_faces:
                flipped_faces.append(cv2.flip(face, 1))
            embs = self.facenet_model_.generate_embeddings_from_images(aligned_faces)
            embeddings.extend(embs)
        flipped_embs = self.facenet_model_.generate_embeddings_from_images(flipped_faces)
        embeddings.extend(flipped_embs)

        return self.clusterer_.score(embeddings, labels)

    def _plot_faces(self, face_paths):
        dim = int(np.sqrt(len(face_paths))) + 1
        fig, axes = plt.subplots(nrows=dim, ncols=dim, figsize=(18, 18))
        for i, ax in enumerate(axes.flatten()):
            if i >= len(face_paths):
                break
            ax.imshow(self._read_rgb_image(face_paths[i]))
        return fig

    def __enter__(self):

        """Returns model when object enters a "with" block"""

        return self

    def __exit__(self):
        self.facenet_model_.close()

    @staticmethod
    def draw_tsne(embs, labels, face_folder):
        transformed = TSNE(n_components=2).fit_transform(embs)
        labels_set = np.unique(labels)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Transformed dim 1', fontsize=15)
        ax.set_ylabel('Transformed dim 2', fontsize=15)
        ax.set_title('2D TSNE', fontsize=20)
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14),
                     (255, 187, 120), (44, 160, 44), (152, 223, 138),
                     (214, 39, 40), (255, 152, 150), (148, 103, 189),
                     (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127),
                     (199, 199, 199), (188, 189, 34), (219, 219, 141),
                     (23, 190, 207), (158, 218, 229)]
        colors = [(i[0] / 255., i[1] / 255., i[2] / 255.)
                  for i in tableau20][: len(labels_set)]

        for label, color in zip(labels_set, colors):
            indices_to_keep = np.where(labels == label)[0]
            samples = transformed[indices_to_keep]
            ax.scatter(samples[:, 0], samples[:, 1], color=color, label=label)

        ax.legend(labels_set)
        ax.grid()
        fig.savefig(os.path.join(face_folder, 'tsne.png'))

    @staticmethod
    def _read_rgb_frame(vidcap):
        ret, frame = vidcap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None

    @staticmethod
    def _get_vid_specs(vidcap):
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fc = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        return width, height, fps, fc

    @staticmethod
    def _read_rgb_image(imag_path):
        image = cv2.imread(imag_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _write_bgr_image(image, path):
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
