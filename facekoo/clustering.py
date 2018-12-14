# -*- coding: utf-8 -*-
"""Randk-Order-Based clustering algorithm.

I combine Approximate-Rank-Order algorithm and Chinese-Whispers algorithm
 in ROCWClustering class.
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors


class BaseClustering:
    def __init__(self):
        pass

    def fit_predict(self, X):
        pass

    def score(self, X, y_true):
        y_pred = self.fit_predict(X)
        pairwise_precision = self._calc_pw_precision(y_pred, y_true)
        pairwise_recall = self._calc_pw_recall(y_pred, y_true)
        pairwise_f_measure = 2 * (pairwise_precision * pairwise_recall)\
            / (pairwise_precision + pairwise_recall)
        return pairwise_f_measure, pairwise_precision, pairwise_recall

    @staticmethod
    def _calc_pw_precision(y_pred, y_true):
        unique_clusters = np.unique(y_pred)
        n_pairs = 0
        n_same_class_pairs = 0
        for cluster in unique_clusters:
            sample_indices = np.where(y_pred == cluster)[0]
            combs = np.array(list(itertools.combinations(sample_indices, 2)), dtype=np.int64)
            if not np.any(combs):
                continue
            combs_classes = y_true[combs]
            same_class_pairs = np.where(combs_classes[:, 0] == combs_classes[:, 1])[0]
            n_pairs += len(combs)
            n_same_class_pairs += len(same_class_pairs)
        pw_precision = n_same_class_pairs / n_pairs
        return pw_precision

    @staticmethod
    def _calc_pw_recall(y_pred, y_true):
        unique_classes = np.unique(y_true)
        n_pairs = 0
        n_same_cluster_pairs = 0
        for clss in unique_classes:
            sample_indices = np.where(y_true == clss)[0]
            combs = np.array(list(itertools.combinations(sample_indices, 2)), dtype=np.int64)
            if not np.any(combs):
                continue
            combs_clusters = y_pred[combs]
            same_cluster_pairs = np.where(combs_clusters[:, 0] == combs_clusters[:, 1])[0]
            n_pairs += len(combs)
            n_same_cluster_pairs += len(same_cluster_pairs)
        pw_recall = n_same_cluster_pairs / n_pairs
        return pw_recall


class ROCWClustering(BaseClustering):
    """Approximated rank-order clustering implemented using Chinese Whispers algorithm.

    Using rank-order distances generate a graph, and feed this graph to ChineseWhispers
     algorithm for clustering.
    """

    def __init__(self, k=20, metric='euclidean', n_iteration=5, algorithm='ball_tree'):
        self.k = k
        self.metric = metric
        self.n_iteration = n_iteration
        self.knn_algorithm = algorithm

    def fit_predict(self, X):
        graph = ROGraph(self.k, self.metric, algorithm=self.knn_algorithm)
        adjacency_mat = graph.generate_graph(X)
        clusterer = ChineseWhispersClustering(self.n_iteration)
        labels = clusterer.fit_predict(adjacency_mat)
        return labels


class ChineseWhispersClustering:
    def __init__(self, n_iteration=5):
        self.n_iteration = n_iteration
        self.adjacency_mat_ = None
        self.labels_ = None

    def fit_predict(self, adjacency_mat):
        """Fits and returns labels for samples"""

        n_nodes = adjacency_mat.shape[0]
        indices = np.arange(n_nodes)
        labels_mat = np.arange(n_nodes)
        for _ in range(self.n_iteration):
            np.random.shuffle(indices)
            for ind in indices:
                weights = adjacency_mat[ind]
                winner_label = self._find_winner_label(weights, labels_mat)
                labels_mat[ind] = winner_label
        self.adjacency_mat_ = adjacency_mat
        self.labels_ = labels_mat
        return labels_mat

    @staticmethod
    def _find_winner_label(node_weights, labels_mat):
        adjacent_nodes_indices = np.where(node_weights > 0)[0]
        adjacent_nodes_labels = labels_mat[adjacent_nodes_indices]
        unique_labels = np.unique(adjacent_nodes_labels)
        label_weights = np.zeros(len(unique_labels))
        for ind, label in enumerate(unique_labels):
            indices = np.where(adjacent_nodes_labels == label)
            weight = np.sum(node_weights[adjacent_nodes_indices[indices]])
            label_weights[ind] = weight
        winner_label = unique_labels[np.argmax(label_weights)]
        return winner_label


class ROGraph:
    def __init__(self, k, metric, algorithm):
        self.k = k
        self.metric = metric
        self.knn_algorithm = algorithm
        self.adjacency_mat_ = None

    @property
    def adjacency_mat(self):
        return self.adjacency_mat_

    def generate_graph(self, X):
        ordered_distances, order_lists = self._get_knns(X)
        pw_distances = self._generate_normalized_pw_distances(ordered_distances, order_lists)
        adjacency_mat = self._generate_adjacency_mat(pw_distances)
        return adjacency_mat

    def _get_knns(self, X):
        """Generates order lists and absolute distances of k-nearest-neighbors
            for each data point.
        """

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm=self.knn_algorithm,
                                metric=self.metric).fit(X)
        ordered_absolute_distances, order_lists = nbrs.kneighbors(X)
        return ordered_absolute_distances, order_lists

    def _generate_normalized_pw_distances(self, ordered_distances, order_lists):
        n_samples = len(ordered_distances)
        combs = itertools.combinations([i for i in range(n_samples)], 2)
        pw_distances = np.zeros((n_samples, n_samples))
        for ind1, ind2 in combs:
            order_list_1, order_list_2 = order_lists[ind1], order_lists[ind2]
            pw_dist = self._calc_pw_dist(ind1, ind2, order_list_1, order_list_2, ordered_distances)
            pw_distances[ind1, ind2] = pw_dist
        pw_distances = pw_distances / pw_distances.max()
        pw_distances = pw_distances + pw_distances.T
        return pw_distances

    def _generate_adjacency_mat(self, pw_distances):
        adjacency_mat = self._dist2adjacency(pw_distances)
        self.adjacency_mat_ = adjacency_mat
        return adjacency_mat

    @staticmethod
    def _dist2adjacency(distances):
        mask_mat = np.zeros(distances.shape)
        mask_mat[np.where(distances > 0)] = 1
        adjacency_mat = (1 - distances) * mask_mat
        return adjacency_mat

    def _calc_pw_dist(self, ind_a, ind_b, order_list_a, order_list_b, ordered_distances):
        pw_dist = 0.0
        if np.any(order_list_a == order_list_b):
            order_b_in_a, order_a_in_b = self._calc_orders(ind_a, ind_b, order_list_a, order_list_b)
            d_m_ab = self._calc_dm(ind_a, ind_b, order_list_a, order_list_b,
                                   order_b_in_a, order_a_in_b)
            d_m_ba = self._calc_dm(ind_b, ind_a, order_list_b, order_list_a,
                                   order_a_in_b, order_b_in_a)
            pw_dist = (d_m_ab + d_m_ba) / min(order_a_in_b, order_b_in_a)
        return pw_dist

    def _calc_orders(self, ind_a, ind_b, order_list_a, order_list_b):
        order_b_in_a = np.where(order_list_a == ind_b)[0]
        if not order_b_in_a.size:
            order_b_in_a = self.k
        else:
            order_b_in_a = order_b_in_a[0]
        order_a_in_b = np.where(order_list_b == ind_a)[0]
        if not order_a_in_b.size:
            order_a_in_b = self.k
        else:
            order_a_in_b = order_a_in_b[0]
        return order_b_in_a, order_a_in_b

    def _calc_dm(self,
                 ind_a, ind_b, order_list_a, order_list_b,
                 order_b_in_a, ordered_distances):
        dist = 0
        for i in range(min(self.k, order_b_in_a)):
            sample_index = order_list_a[i]
            if np.any(order_list_b == sample_index):
                dist += 1 / self.k
            else:
                dist += 1
        return dist
