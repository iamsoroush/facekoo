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
import re
import logging

import tensorflow as tf
import numpy as np
import skimage.io as skio
import skimage.color as skcolor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import dlib
from collections import OrderedDict
import cv2


class FaceClusering:
    def __init__(self, args):
        self.facenet_model_ = None
        self.clusterer_ = None
        self.face_detector_ = None
        self.face_aligner_ = None

    def do_clustering(self, X):
        """Clustering on given samples.

        Args:
            X: an array of arrays, each row is a sample and columns are
             features.

        Returns:
            an array that contains predicted labels for given X.
        """

        pass

    def do_clustering_on_images(self, images):
        """Clustering on given images.

        Args:
            images: list of RGB images.

        Returns:
            an array that contains predicted labels for given images.
        """

        pass

    def do_clustering_on_image_paths(self, image_paths):
        """Clustering on images found on specified paths.

        Args:
            image_paths: list of complete paths to images

        Returns:
            an array that contains predicted labels for given images.
        """

        pass

    def do_clustering_on_video(self, video_path):
        """Clustering on given video path.

        Args:
            video_path: complete path to video.

        Returns:
            a dictionary containing cropped faces for each cluster:
                {cluster1: [face1, face2, ...], cluster2: [face1, face2], ...}
        """

        pass

    def evaluate(self, X, y_true):
        """Evaluates clustering algorithm on given data.

        Args:
            X: an array of arrays
            y_true: 1D array, true labels for each row of X

        Returns:
            pairwise f-measure
        """

        pass

    def evaluate_on_images(self, images, labels):
        """Evaluates clustering on given images.

        Args:
            images: array of rgb images.
            labels: array of labels for given images.

        Returns:
            pairwise f-measure
        """

        pass

    def __enter__(self):
        """Returns model when object enters a "with" block"""

        return self

    def __exit__(self):
        self.facenet_model_.close()


class FaceNet:
    """FaceNet class containing the facenet model ready to generate embeddings
    for faces.

    Naming rules:
        self.name : externally given parameters (as argument), and public
         parameters
        self.name_ : internally created, non-public parameters
        self.name() : public methods
        self._name() : internal methods

    Public methods:
        generate_embeddings_from_images: given images, generates 512D embeddings.
        generate_embeddings_from_paths: generates embeddings for images found at
            specified paths.
        plot_2d_embeddings: plots generated embeddings for generated embeddings in 2D space,
            after performing dimension reduction using 'TSNE' method.
        clean: closes the running tensorflow Session.

    Attributes:
        sess_: tensorflow session that contains compiled pretrained facenet model.
    """

    def __init__(self, model_name='20180402-114759', batch_size=32,
                 image_size=160, do_prewhiten=True, do_crop=True):
        """Initialize a FaceNet model.

        Args:
            model_name: model files must be on 'models/model_name' directory.
            batch_size: size of batch to pass to tensorflow's session.run() for feature generation.
            image_size: facenet model will use images at this size if do_crop=True, all training and
                         testing images will be transformed to this size before feature generation.
                        pass aligned faces in 182*182 size to the model and set this parameter to
                         160 for good performance.
            do_prewhiten: Whether to do prewhiten before generating embeddings or not.
            do_crop: Whether or not to do crop on images before generating features. Cropped
                      images will be the (image_size, image_size) shaped region on the centre
                      of original image.
        """

        self.logger_ = self._initialize_logger()
        self.model_dir_ = os.path.join(os.path.dirname(__file__) + '/models/', model_name)
        self.logger_.info('Model dir: {}'.format(self.model_dir_))
        self.sess_, self.graph_ = self._load_model()
        self.closed_ = False
        (self.images_placeholder_,
         self.embeddings_tensor_,
         self.phase_train_placeholder_) = self._get_tensors()
        self.batch_size = batch_size
        self.image_size = image_size
        self.do_prewhiten = do_prewhiten
        self.do_crop = do_crop

    def _initialize_logger(self):
        """Initializes a console logger for logging purposes."""

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger = logging.getLogger('FaceNet')
        logger.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        return logger

    def _load_model(self):
        """Loads and compiles pretrained model's graph and associated tensors.

        Returns:
            a tensorflow session that contains the loaded and compiled model.

        Raises:
            ValueError: If pretrained model not found or more than one model exists
                in self.model_dir_.
        """

        try:
            meta_file, ckpt_file = self._get_model_filenames(self.model_dir_)
        except Exception as e:
            self.logger_.error(e)
            raise e

        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(self.model_dir_, meta_file), input_map=None)
        saver.restore(sess, os.path.join(self.model_dir_, ckpt_file))
        self.logger_.info("Model loaded. Session is ready.")
        return sess, sess.graph

    def _get_tensors(self):
        """Returns needed tensors: placeholders and embeddings tensor."""

        images_placeholder = self.graph_.get_tensor_by_name("input:0")
        embeddings = self.graph_.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph_.get_tensor_by_name("phase_train:0")

        return images_placeholder, embeddings, phase_train_placeholder

    @staticmethod
    def _get_model_filenames(model_dir):
        """Searches specified paths for model files.

        Returns:
            Model's meta file and checkpoint file

        Raises:
            ValueError: If model's meta file not found or more than one meta file exists.
        """

        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in\
             the model directory (%s)'
                             % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    def generate_embeddings_from_images(self, images):
        """Generate embeddings for given images.

        Args:
            images: array of RGB images.

        Returns:
            array of 512-dimensional embeddings generated for given images.
        """

        preprocessed_images = self._preprocess_images(images)
        emb_array = self._generate_embeddings(preprocessed_images)

        return emb_array

    def generate_embeddings_from_paths(self, image_paths):
        """Generate embeddings for images found in specified paths.

        Args:
            image_paths: list of complete paths to images.

        Returns:
            generated 512-dimensional embeddings for images found on paths.
        """

        embedding_size = self.embeddings_tensor_.get_shape()[1]

        # Run forward pass to calculate embeddings
        self.logger_.info('Calculating embeddings ...')
        nrof_images = len(image_paths)
        nrof_batches_per_epoch = int(np.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))

        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = image_paths[start_index: end_index]
            images = self._load_data_from_paths(paths_batch)
            emb_array[start_index: end_index, :] = self._generate_batch_embeddings(images)

        return emb_array

    def plot_2d_embeddings(self, embs, labels):
        """Plots embeddings after dimension reduction.

        For explatory tasks only.

        Args:
            embs: generated embeddings for faces.
            labels: labels for given embeddings.
        """

        transformed = TSNE(n_components=2).fit_transform(embs)
        labels_set = np.unique(labels)

        fig = plt.figure(figsize=(8, 8))
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
        plt.show()
        return

    def clean(self):
        """Closes background tensorflow session."""

        self.sess_.close()
        self.closed_ = True

    def _generate_embeddings(self, images):
        """Generates embeddings for given images.

        Args:
            images: preprocessed images.

        Returns:
            generated embeddings.
        """

        embedding_size = self.embeddings_tensor_.get_shape()[1]

        # Run forward pass to calculate embeddings
        nrof_images = images.shape[0]
        nrof_batches_per_epoch = int(np.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))

        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            imgs = images[start_index: end_index]
            emb_array[start_index: end_index, :] = self._generate_batch_embeddings(imgs)

        return emb_array

    def _generate_batch_embeddings(self, batch):
        """One run over a batch, that generates embeddings for given batch.

        Args:
            batch: array of preprocessed images.

        Returns:
            generated embeddings
        """

        feed_dict = {self.images_placeholder_: batch, self.phase_train_placeholder_: False}
        return self.sess_.run(self.embeddings_tensor_, feed_dict=feed_dict)

    def _load_data_from_paths(self, image_paths):
        """Loads images in RGB, and returns preprocessed images.

        Args:
            image_paths: [path1, path2, ...]
            Note: images must be aligned faces.

        Returns:
            preprocessed images.
        """

        nrof_samples = len(image_paths)
        images = np.zeros((nrof_samples, self.image_size, self.image_size, 3))

        for i in range(nrof_samples):
            image_path = os.path.expanduser(image_paths[i])
            img = skio.imread(image_path)
            if not np.any(img):
                print('image not found.')
                continue
            img = self._preprocess_image(image=img)
            images[i, :, :, :] = img
        return images

    def _preprocess_images(self, images):
        """Preprocesses array of images.

        Note: Pass aligned faces of size (182, 182) to this method


         if self.image_size is (160, 160)
         for performance maintaining.

        Args:
            images: array of RGB images.

        Returns:
            preprocessed images: cropped and prewhitened.
        """

        nrof_samples = len(images)
        imgs = np.zeros((nrof_samples, self.image_size, self.image_size, 3))
        for i, img in enumerate(images):
            img = self._preprocess_image(image=img)
            imgs[i, :, :, :] = img
        return imgs

    def _preprocess_image(self, image):
        """Preprocess image for feeding to model.

        Note: Pass aligned face of size (182, 182) to this method if self.image_size is (160, 160)
         for performance maintaining.

        Args:
            image: RGB image to preprocess.

        Returns:
            prewhitened and cropped image.
        """

        if self.do_prewhiten:
            image = self._prewhiten(image)
        if self.do_crop:
            image = self._crop(image, self.image_size)
        return image

    @staticmethod
    def _prewhiten(x):
        """Input image standardization.

        This makes predictions much accurate than using non-whitened images.

        Args:
            x: input image

        Returns:
            whitened image
        """

        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    @staticmethod
    def _crop(image, image_size):
        """Crops given image and returns central (image_size, image_size) region."""

        if image.shape[1] > image_size:
            sz1 = int(image.shape[1] // 2)
            sz2 = int(image_size // 2)
            (h, v) = (0, 0)
            image = image[(sz1 - sz2 + v):(sz1 + sz2 + v),
                          (sz1 - sz2 + h):(sz1 + sz2 + h), :]
        return image


class FaceDetector:
    """A class for detecting faces in images or frames.

    Give image to 'detect_faces' method, and it will return all faces as an array.

    Public methods:
        detect_faces: detects faces in given image.

    Attributes:
        detector_: dlib's HOG-based frontal face detector.
    """

    def __init__(self):
        self.detector_ = dlib.get_frontal_face_detector()

    def detect_faces(self, gray_image):
        """Detects faces in given image.

        Args:
            image: gray-scale image

        Returns:
            detected faces: an array of rectangles.
        """

        # resized_image = self._resize_image(image, out_pixels_wide=800)
        # gray = skcolor.rgb2rgb2gray(resized_image)
        rects = self.detector_(gray_image, 0)  # Detect faces in the gray scale frame
        return rects

    @staticmethod
    def _resize_image(img, out_pixels_wide):
        ratio = out_pixels_wide / img.shape[1]
        dim = (int(out_pixels_wide), int(img.shape[0] * ratio))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized


class FaceAligner:
    """Aligns faces, use aligned faces for generating embeddings via FaceNet.

    Public methods:
        align: aligns faces found in given image.

    Attributes:
        predictor_: dlib's facial shape predictor (5 landmarks).
        landmarks_idxs_: Positions for output landmarks.
        desired_left_eye: Desired output left eye position, given in a (x, y) tuple.
            Percentages between (0.2, 0.4), With 20% you’ll basically be getting a
            “zoomed in” view of the face, whereas with larger values the face will
            appear more “zoomed out.”
        desired_face_width: Output images width, in pixels.
    """

    def __init__(self,
                 predictor_name='shape_predictor_5_face_landmarks.dat',
                 desired_left_eye=(0.35, 0.35),
                 desired_face_width=182):
        """Init a FaceAligner instance.

        Args:
            predictor_name: name of dlib's facial shape predictor (5 landmarks) located in\
             models/shape_predictor/
            desired_left_eye: Desired output left eye position, given in a (x, y) tuple.
                Percentages between (0.2, 0.4), With 20% you’ll basically be getting a
                “zoomed in” view of the face, whereas with larger values the face will
                appear more “zoomed out.”
            desired_face_width: Output images width, in pixels.
        """

        predictor_dir = os.path.join(os.path.dirname(__file__) + '/models/shape_predictor/',
                                     predictor_name)
        self.predictor_ = dlib.shape_predictor(predictor_dir)
        FACIAL_LANDMARKS_5_IDXS = OrderedDict([
            ("right_eye", (2, 3)),
            ("left_eye", (0, 1)),
            ("nose", (4,))
        ])
        self.landmarks_idxs_ = FACIAL_LANDMARKS_5_IDXS
        self.desired_left_eye = desired_left_eye
        self.desired_right_eye = (1.0 - self.desired_left_eye[0], self.desired_left_eye[1])
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_width

    def align(self, image, rects):
        """Aligns the faces specified by rects in image.

        Args:
            image: The RGB input image.
            rects: The bounding box rectangles produced by dlib’s HOG face detector.

        Returns:
            a list of aligned faces.
        """

        gray = skcolor.rgb2gray(image).astype(np.uint8)
        outputs = list()
        for rect in rects:
            shape = self.predictor_(gray, rect)
            predicted_coords = self._shape_to_np(shape)

            tform_mat = self._calc_transform_mat(predicted_coords)

            output = self._apply_transformation(image, tform_mat)
            outputs.append(output)

        return outputs

    def _apply_transformation(self, image, M):
        """Applies affine transform to given image using given matrix.

        Args:
            image: RGB image
            M: transformation matrix

        Returns:
            Aligned face
        """

        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        return output

    def _calc_transform_mat(self, predicted_coords):
        """Generates transorfmation matrix.

        Args:
            predicted_coords: predicted landmarks' coordinates

        Returns:
            Transorm matrix, should given to warpAffine
        """

        eyes_center, angle, scale = self._calc_mat_params(predicted_coords)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        return M

    def _calc_mat_params(self, predicted_coords):
        (left_eye_start, left_eye_end) = self.landmarks_idxs_["left_eye"]
        (right_eye_start, right_eye_end) = self.landmarks_idxs_["right_eye"]

        left_eye_pts = predicted_coords[left_eye_start: left_eye_end + 1]
        right_eye_pts = predicted_coords[right_eye_start: right_eye_end + 1]

        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        desired_right_eye_x = 1.0 - self.desired_left_eye[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist

        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
        return eyes_center, angle, scale

    @staticmethod
    def _shape_to_np(shape, dtype="int"):

        # Initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords
