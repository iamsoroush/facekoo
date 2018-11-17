# -*- coding: utf-8 -*-
"""Face Recognizer.

   This file contains FaceRecognizer class.
   It uses facenet's pretrained CNN.

   Create an instance of this class, train the model on your data by calling the
    'train' method, and predict incoming new face's label via 'predict' method.

   You can use the model in context manager mode, but the session will terminate
    at the end of your 'with' block. When using in standard way, the background
    tensorflow session wouldn't terminate in order to avoid redundant operations.
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math
import pickle
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import logging
import re
import imgaug.augmenters as iaa
import outlierdetection


class FaceRecognizer:
    """FaceRecognizer initializer.

    The pretrained model is encapsulated in this class.

    Naming rules:
        self.name : externally given parameters (as argument), and public parameters
        self.name_ : internally created, non-public parameters
        self.name() : public methods
        self._name() : internal methods

    Public attributes:
        random_seed: random_seed for random numpy and tensorflow operations.
        model_dir: Directory of pretrained facenet model.
        batch_size: Size of batch to pass to tensorflow session.run for feature generation
        image_size: Facenet model will use images at this size if do_crop=True, all training and
                     testing images will be transformed to this size before feature generation.
                    Give aligned images in 182*182 size to the model and set this parameter to 160
                     for good performance.
        embeddings_filename: Name of previously saved embeddings, or file name for
                              saving incoming new embeddings.
        classifier_filename_exp: Directory of saved classifier, or directory for saving
                                  new classifer.
        train_dir: Directory of train data for initial training. Each training class must be
                    in a subdirectory on this directory.
        do_prewhiten: Whether to do prewhiten before generating embeddings or not.
        do_crop: Whether or not to do crop on images before generating features. Cropped
                  images will be the centered (image_size, image_size) of original image.
        min_augmented_images_per_class: Before training the classifier, images will augment
                                         to satisfy this number of images per class.

    Public methods:
        train: Use this method for training the model.
        retrain: Retrains the model on existing training data and new given data.
                  Use this method for online training.
        predict: Predicts the label for given image(s).
        plot_2d_embeddings: Plots data in 2d space using specified method.
        clean: Use this method when the whole program is exitting. This method calls
                'close' method of tensorflow session.

    """

    def __init__(self, random_seed=666, model_dir='models/20180402-114759',
                 batch_size=32, image_size=160, classifier_filename='classifier.pkl',
                 embeddings_filename='embeddings.pkl', train_dir='dataset/train',
                 do_prewhiten=True, do_crop=True, min_augmented_images_per_class=100):
        """Inits a FaceRecognizer object.

        self.sess_ : tensorflow session that contains loaded model's graph
        self.embeddings_ : {'class1': [sample1_embedding, ...], ... , 'classn': [sample1_embedding, ...]}
        self.classifier_ : (classifier object, class names)
        """

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger_ = logging.getLogger('FaceRecognizer')
        self.logger_.setLevel(logging.INFO)
        self.logger_.addHandler(console_handler)
        self.random_seed = random_seed
        self.model_dir = model_dir
        self.sess_ = self._load_model()
        self.batch_size = batch_size
        self.image_size = image_size
        self.embeddings_filename = embeddings_filename
        self.embeddings_ = self._load_embeddings()
        self.classifier_filename_exp = os.path.expanduser(classifier_filename)
        self.classifier_ = self._load_classifier()
        self.train_dir = train_dir
        self.do_prewhiten = do_prewhiten
        self.do_crop = do_crop
        self.min_augmented_images_per_class = min_augmented_images_per_class

    def __enter__(self):
        """Returns model when object enters a "with" block"""

        return self.model

    def __exit__(self, ctx_type, ctx_value, ctx_traceback):
        self.sess_.close()

    @property
    def embeddings_(self):
        return self.embeddings__

    @embeddings_.setter
    def embeddings_(self, new_dict):
        with open(self.embeddings_filename, 'wb') as file:
            pickle.dump(new_dict, file)
            self.embeddings__ = new_dict

    @property
    def classifier_(self):
        return self.classifier__

    @classifier_.setter
    def classifier_(self, new_classifier):
        with open(self.classifier_filename_exp, 'wb') as outfile:
            pickle.dump(new_classifier, outfile)
            self.classifier__ = new_classifier

    def _load_embeddings(self):
        """Loads early generated embeddings.

        Returns:
            "None" if self.embeddings_filename path does not exist,
            retuens saved embeddings else
        """
        if os.path.exists(self.embeddings_filename):
            with open(self.embeddings_filename, 'rb') as file:
                embeddings = pickle.load(file)
            self.logger_.info('Embeddings file loaded.')
        else:
            embeddings = None
            self.logger_.warning('Embeddings file not found.')
        return embeddings

    def _load_classifier(self):
        """Loads saved classifer and class names.

        self.classifier_filename_exp must contain a tuple:
         (classifier object, class names)

        Returns:
            (None, None) if self.classifier_filename_exp path does not exist
            (classifier object, class names) else
        """
        if os.path.exists(self.classifier_filename_exp):
            with open(self.classifier_filename_exp, 'rb') as file:
                classifier, class_names = pickle.load(file)
            self.logger_.info('Classifier loaded.')
        else:
            classifier, class_names = None, None
            self.logger_.warning('Classifier not found.')
        return classifier, class_names

    def _load_model(self):
        """Loads pretrained model's graph and associated tensors.

        Returns:
            a tensorflow session that contains the loaded model

        Raises:
            ValueError: If pretrained model not found or more than one model exists
                in specified path.
        """

        model_exp = os.path.expanduser(self.model_dir)
        try:
            meta_file, ckpt_file = self._get_model_filenames(model_exp)
        except Exception as e:
            self.logger_.error(e)
            raise e

        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=None)
        saver.restore(sess, os.path.join(model_exp, ckpt_file))
        self.logger_.info("Model loaded. Session is ready.")
        return sess

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
            raise ValueError('There should not be more than one meta file in the model directory (%s)'
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

    def train(self):
        """Trains model on augmented data and tests on training data.

        Loads training data from self.train_dir, augments loaded data and fits a SVM
         classifier on data, updates classifer and embeddings files and prints out accuracy
         on unaugmented data.
        """

        self.logger_.info('Starting to train ...')
        dataset = self._get_dataset()
        for cls in dataset:  # Check that there are at least five training image per class
            assert len(cls.image_paths) > 5, 'There must be at least ten image for each class in the dataset'
        paths, labels = self._get_image_paths_and_labels(dataset)
        class_names = [cls.name.replace('_', ' ') for cls in dataset]  # Create a list of class names
        lbls = [class_names[lbl] for lbl in labels]  # Contains the names of labels
        self.logger_.info('Number of training classes: %d' % len(dataset))
        self.logger_.info('Number of training images: %d' % len(paths))
        self.logger_.info('Training classifier ...')
        augmented_embs, augmented_labels = self._generate_augmented_embeddings(paths, labels)
        model = SVC(kernel='linear', probability=True)  # Train classifier
        model.fit(augmented_embs, augmented_labels)
        self.logger_.info('Classifier score: {}'.format(model.score(augmented_embs, augmented_labels)))
        self.classifier = (model, class_names)

        # Testing classifier on training data
        self.logger_.info('Testing classifier on training data ...')
        emb_array = self._generate_embeddings_from_paths(paths=paths)  # Calculate embeddings for train data
        embedding_dict = {p: list() for p in class_names}
        for ind, embedding in enumerate(emb_array):
            embedding_dict[lbls[ind]].append(embedding)
        self.embeddings = embedding_dict
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f label: %s' % (i, class_names[best_class_indices[i]],
                                               best_class_probabilities[i], lbls[i]))
        accuracy = np.mean(np.equal(best_class_indices, labels))
        self.logger_.info('Accuracy: %.3f' % accuracy)
        self.logger_.info('Verification score: %.3f +- %.3f' % (best_class_probabilities.mean(),
                                                                best_class_probabilities.std()))
        return

    def _get_dataset(self):
        """Returns list of ImageClass objects, each object belongs to one class."""

        path = self.train_dir
        dataset = []
        path_exp = os.path.expanduser(path)
        classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = self._get_image_paths(facedir)
            dataset.append(self.ImageClass(class_name, image_paths))

        return dataset

    @staticmethod
    def _get_image_paths(facedir):
        """Returns paths of images in class directory 'facedir' ."""

        image_paths = []
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_paths = [os.path.join(facedir, img) for img in images]
        return image_paths

    class ImageClass:
        """Stores the paths to images for a given class"""

        def __init__(self, name, image_paths):
            self.name = name
            self.image_paths = image_paths

        def __str__(self):
            return self.name + ', ' + str(len(self.image_paths)) + ' images'

        def __len__(self):
            return len(self.image_paths)

    @staticmethod
    def _get_image_paths_and_labels(dataset):
        """Returns paths and labels of all images in training dataset.

        Args:
            dataset: list of ImageClass objects.

        Returns:
            (image paths, labels) in flatten format
        """

        image_paths_flat = []
        labels_flat = []
        for i in range(len(dataset)):
            image_paths_flat += dataset[i].image_paths
            labels_flat += [i] * len(dataset[i].image_paths)
        return image_paths_flat, labels_flat

    def _generate_augmented_embeddings(self, paths, labels):
        """Loads images from paths, augments images and generates embeddings.

        Args:
            paths: paths to images [[path1, path2, ...], [path1, path2, ...], ...]
            labels: labels for images, same length as paths

        Returns:
            (embeddeings for augmented images, corresponding labels)
        """

        self.logger_.info("Generating embeddings for augmented version of training data ...")
        zipped = list(zip(paths, labels))
        augmented_embeddings = list()
        augmented_labels = list()
        for label in set(labels):
            img_paths = [i[0] for i in zipped if i[1] == label]
            noi = len(img_paths)
            if noi < self.min_augmented_images_per_class:
                random_indices = list(np.random.randint(0, noi, self.min_augmented_images_per_class - noi))
                copies = [img_paths[i] for i in random_indices]
                img_paths.extend(copies)
            images = self._augment_images(img_paths)
            embs = self._generate_embeddings_from_images(images)
            augmented_embeddings.extend(embs)
            augmented_labels.extend([label] * len(img_paths))
        return np.array(augmented_embeddings), augmented_labels

    def _augment_images(self, paths):
        """Augments images that exist in given path.

        Augmentation will perfoem in this manner:
            flip: 50 percent of images will be flipped
            # crop: crops and pads a random 10% of pixels
            # sometimes gaussian blur: Small gaussian blur with random sigma between 0 and 0.5.
             on about 50% of all images
            affine: applies moderate random affine transformations

        Returns:
            Augmented images
        """

        seq = iaa.Sequential([iaa.Fliplr(0.5),  # horizontal flips
                              # iaa.Crop(percent=(0, 0.1)),  # random crops
                              # Small gaussian blur with random sigma between 0 and 0.5.
                              # But we only blur about 50% of all images.
                              iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),
                              # Apply affine transformations to each image.
                              # Scale/zoom them, translate/move them, rotate them and shear them.
                              iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                         rotate=(-10, 10),
                                         shear=(-4, 4))
                              ],
                             random_order=True)  # apply augmenters in random order
        images = self._load_data_from_paths(paths)
        augmented_images = seq.augment_images(images)
        return augmented_images

    def _generate_embeddings_from_images(self, images):
        """Generates embeddings for given images.

        Args:
            images: preprocessed images

        Returns:
            generated embeddings
        """

        # np.random.seed(seed=self.random_seed)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        nrof_images = images.shape[0]
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            imgs = images[start_index: end_index]
            feed_dict = {images_placeholder: imgs, phase_train_placeholder: False}
            emb_array[start_index: end_index, :] = self.sess_.run(embeddings, feed_dict=feed_dict)
        return emb_array

    def _generate_embeddings_from_paths(self, paths):
        """Generates embeddings for images found on given paths.

        Args:
            paths: [[c1_path1, c1_path2, ...], [c2_path1, c2_path2, ...], ...]

        Returns:
            generated embeddings
        """
        # np.random.seed(seed=self.random_seed)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        self.logger_.info('Calculating embeddings ...')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = paths[start_index: end_index]
            images = self._load_data_from_paths(paths_batch)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index: end_index, :] = self.sess_.run(embeddings, feed_dict=feed_dict)
        return emb_array

    def _generate_embedding_from_image(self, img):
        """Generates embedding for a single image.

        Args:
            img: must be of type 'RGB'

        Returns:
            generated embedding (embedding_size,)
        """
        img = self._preprocess_image(img)
        image = img[np.newaxis, :]
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        # self.logger_.info('Calculating embedding for image ...')
        emb_array = np.zeros((1, embedding_size))
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        emb_array[0, :] = self.sess_.run(embeddings, feed_dict=feed_dict)
        return emb_array[0]

    def retrain(self, images, class_name, is_bgr=True):
        """Retrains the classifier after adding new images to the dataset.


        Args:
            images: array of images of format RGB, BGR, grayscale
            class_name: class name for images
            is_bgr: are images loaded in BGR format?
        """

        class_dir = self.train_dir + '/' + class_name
        nof_images = 0
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        else:
            nof_images = len(os.listdir(class_dir))

        imgs = self._preprocess_images(images, is_bgr=is_bgr)
        embs = self._generate_embeddings_from_images(imgs)
        # inliers_labels = np.where(self.detect_outliers(embs) == 1)[0]
        # outliers_n = len(images) - len(inliers_labels)
        # images = images[inliers_labels]
        # self.logger_.info("{} outliers found and filtered.".format(outliers_n))
        for ind, image in enumerate(images):
            cv2.imwrite(class_dir + '/' + str(nof_images + ind + 1) + '.jpg',
                        image)
        self.train()
        return

    def predict(self, img, is_bgr=True):
        """Predicts input image's class.

        Args:
            img: given image, must be of format RGB, BGR
            is_bgr: True if the given image is of format BGR

        Returns:
            (False, False) if classifier not found,
            (predicted class name, associated probability) else
        """

        if self.classifier is None:
            self.logger_.warning('Classifier not found.')
            return False, False
        if is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model, class_names = self.classifier
        embedding = self._generate_embedding_from_image(img)
        prediction = model.predict_proba(embedding[np.newaxis, :])[0]
        best_class_index = np.argmax(prediction)
        class_name = class_names[best_class_index]
        class_prob = prediction[best_class_index]
        return class_name, class_prob

    def plot_2d_embeddings(self, technique='tsne'):
        """Plots embeddings after dimension reduction.

        For explatory tasks only.

        Args:
            technique: 'tsne' or 'pca'
        """

        if not self.embeddings:
            print('Embedding dictionary pickle not found.')
            return
        embeddings = list()
        labels = list()
        for key, value in self.embeddings.items():
            for embedding in value:
                embeddings.append(embedding)
                labels.append(key)
        labels_set = list(set(labels))
        if technique == 'tsne':
            transformed = TSNE(n_components=2).fit_transform(embeddings)
        elif technique == 'pca':
            transformed = PCA(n_components=2).fit_transform(embeddings)
        principal_df = pd.DataFrame(data=transformed,
                                    columns=['principal component 1',
                                             'principal component 2'])
        final_df = pd.concat([principal_df, pd.DataFrame({'label': labels})],
                             axis=1)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
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
            indices_to_keep = final_df['label'] == label
            ax.scatter(final_df.loc[indices_to_keep, 'principal component 1'],
                       final_df.loc[indices_to_keep, 'principal component 2'],
                       color=color, label=label)
        ax.legend(labels_set)
        ax.grid()
        plt.show()
        return

    def clean(self):
        """Closes running tensorflow session.

        Call this method on exit.
        """

        self.sess_.close()
        return

    def _load_data_from_paths(self, image_paths):
        """Loads images in RGB format, and preprocesses the loaded images.

        Args:
            image_paths: [[c1_path1, c1_path2, ...], [c2_path1, c2_path2, ...], ...]

        Returns:
            loades images
        """

        nrof_samples = len(image_paths)
        images = np.zeros((nrof_samples, self.image_size, self.image_size, 3))
        for i in range(nrof_samples):
            img = cv2.imread(image_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._preprocess_image(image=img)
            images[i, :, :, :] = img
        return images

    def _preprocess_image(self, image):
        """Preprocess image for feeding to model.

        Args:
            image: must be of format grayscale or RGB
            image_size: desired image size for feeding to model

        Returns:
            preprocessed image
        """

        if image.ndim == 2:
            image = self._to_rgb(image)
        if self.do_prewhiten:
            image = self._prewhiten(image)
        if self.do_crop:
            image = self._crop(image, self.image_size)
        return image

    def _preprocess_images(self, images, is_bgr=True):
        """Preprocesses array of images.

        Args:
            images: array of images of formats RGB, BGR, grayscale
            is_bgr: are the images loaded in bgr format

        Returns:
            preprocessed images
        """

        nrof_samples = len(images)
        imgs = np.zeros((nrof_samples, self.image_size, self.image_size, 3))
        for i, img in enumerate(images):
            if is_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._preprocess_image(image=img)
            imgs[i, :, :, :] = img
        return imgs

    @staticmethod
    def _prewhiten(x):
        """Input image standardization.

        This makes predictions much accurate than when train and test on non-whitened images.

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

    @staticmethod
    def _to_rgb(img):
        """Transforms grayscale image to RGB.

        Args:
            img: must be of size 2

        Returns:
            given image in RGB format
        """

        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    @staticmethod
    def detect_outliers(embeddings, method='hillout'):
        outlier_detector = outlierdetection.HilOutOD(n_neighbors=int(len(embeddings)/2),
                                                     outlier_constant=1.5, algorithm='brute',
                                                     distance_metric='cosine')
        labels = outlier_detector.fit_predict(embeddings)
        return labels


if __name__ == '__main__':
    fr = FaceRecognizer(do_prewhiten=True, do_crop=True)
    fr.train()
    fr.plot_2d_embeddings(technique='tsne')
    fr.clean()
