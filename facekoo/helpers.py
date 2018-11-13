import numpy as np
import cv2
import os
from sklearn import metrics


def clustering_evaluation(data, labels, model, n_iterations=50):
    """I will use v_measure from sklearn for evaluating. It is same as pair-wise precision and recall.\

    homogeneity (h) : each cluster contains only members of a single class.\
    completeness (c) : all members of a given class are assigned to the same cluster.\
    v_measure (v) = 2 * (h * c)/(h + c)

    Parameters
    ----------
    data: :type: np.ndarray of np.ndarray objects.
        Each array represents features extracted from a face image.

    labels: :type: np.ndarray of int objects.
        Labels for embeddings given in data.

    model: clustering model

    n_iterations: int
        Iterations for calculating score on random sampled data.


    returns: mean(v) - std(v), mean(h) - std(h)

    """

    def gen_rand_data(data, labels, proportion):
        data_len = data.shape[0]

        # Generate random indices without replacement
        indices = np.arange(data_len)
        np.random.shuffle(indices)

        # Select specified proportion of data
        n_samples = int(proportion * data_len)
        indices_to_select = indices[: n_samples]

        return data[indices_to_select], labels[indices_to_select]

    # Initialize the data holders
    v_measures = np.zeros(n_iterations)
    homogeneities = np.zeros(n_iterations)
    completenesses = np.zeros(n_iterations)

    # Evaluate clustering model on random subsets of data
    for i in range(n_iterations):

        # Generate random subset from data
        rand_data, rand_labels = gen_rand_data(data, labels, proportion=0.8)

        # Perform clustering on subset data
        predicted_labels = model.fit_predict(rand_data)

        # Calculate measures
        h, c, v = metrics.homogeneity_completeness_v_measure(rand_labels, predicted_labels)

        v_measures[i] = v
        homogeneities[i] = h
        completenesses[i] = c
        print('Iteration number {} ended.'.format(i + 1))

    # Calculate scores
    score_v = v_measures.mean() - v_measures.std()
    score_h = homogeneities.mean() - homogeneities.std()

    return score_v, score_h


def create_random_dataset(data_dir, inliers_n=50, outliers_n=10, n_samples=30):
    classes = os.listdir(data_dir)
    np.random.shuffle(classes)
    selected_class = classes[0]
    class_dir = data_dir + '/' + selected_class
    img_dirs = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
    np.random.shuffle(img_dirs)
    img_dirs = img_dirs[: inliers_n]
    labels = [selected_class] * inliers_n
    for cls in classes[: -1]:
        cls_dir = data_dir + '/' + cls
        dirs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)]
        np.random.shuffle(dirs)
        img_dirs.extend(dirs[: outliers_n])
        labels.extend([cls] * outliers_n)
    random_indices = np.random.randint(0, len(img_dirs), n_samples)
    img_dirs = np.array(img_dirs)[random_indices].tolist()
    labels = np.array(labels)[random_indices].tolist()
    return img_dirs, selected_class, labels


def drop_high_correlated_images(flatten_df, images, threshold=0.9):
    corr_matrix = flatten_df.T.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return np.delete(images, tuple(to_drop), axis=0)


def one_to_many_high_correlated_indices(new_flatten_gray_images, saved_gray_image, threshold=0.9):
    images_arr = np.append(new_flatten_gray_images, [saved_gray_image], axis=0)
    corr = np.corrcoef(images_arr)[-1][: -1]
    return corr < threshold


def draw_border(img, pt1, pt2, color, thickness, r, d, label='Unknown'):
    """Fancy box drawing function for detected faces."""

    x1, y1 = pt1
    x2, y2 = pt2

    # Top left drawing
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right drawing
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left drawing
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right drawing
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # Writing image's label
    cv2.putText(img=img, text=label, org=(x1 + r, y1 - 3*r), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=color, lineType=cv2.LINE_AA)


def resize_image(img, out_pixels_wide=800):
    ratio = out_pixels_wide / img.shape[1]
    dim = (int(out_pixels_wide), int(img.shape[0] * ratio))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h
