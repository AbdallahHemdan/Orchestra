from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn import svm, tree
import numpy as np
import argparse
import imutils
import cv2
import os
import skimage.io as io
import random
from skimage.transform import rotate
from sklearn.model_selection import train_test_split


########## Variables ##########

random_seed = 42
random.seed(random_seed)
target_img_size = (32, 32)
np.random.seed(random_seed)

classifiers = {
    'SVM': svm.LinearSVC(random_state=random_seed),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'NN': MLPClassifier(solver='sgd', random_state=random_seed, hidden_layer_sizes=(500,), max_iter=100, verbose=1),
    'TREE': tree.DecisionTreeClassifier(random_state=0, max_depth=10)
}

########## Methods ##########


def extract_raw_pixels(img):
    return cv2.resize(img, target_img_size).flatten()


def extract_sift_features(img):
    img = cv2.resize(img, target_img_size)

    sift = cv2.SIFT_create()
    _, features = sift.detectAndCompute(img, None)

    try:
        return features.flatten()
    except:
        io.imshow(img)
        io.show()


def extract_hog_features(img):
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def extract_features(img, feature_set='hog'):
    if feature_set == 'hog':
        return extract_hog_features(img)
    elif feature_set == 'sift':
        return extract_sift_features(img)
    else:
        return extract_raw_pixels(img)


def get_directories():
    directories = []
    directories_filenames = os.listdir('./data-set')

    for i, fn in enumerate(directories_filenames):
        directories.append(fn)

    return directories


def load_dataset(feature_set='hog'):
    labels = []
    features = []
    directories = get_directories()

    for dir_name in directories:
        path_to_dataset = './data-set/' + dir_name
        img_filenames = os.listdir(path_to_dataset)

        for i, fn in enumerate(img_filenames):
            label = dir_name
            labels.append(label)

            path = os.path.join(path_to_dataset, fn)
            img = cv2.imread(path)
            features.append(extract_features(img, feature_set))

        print('finished processing: ', dir_name)

    return features, labels


def run_experiment(train_features, test_features, train_labels, test_labels, model_name):
    model = classifiers[model_name]
    print('############## Training', model_name, "##############")
    # Train the model only on the training features
    model.fit(train_features, train_labels)

    # Test the model on images it hasn't seen before
    accuracy = model.score(test_features, test_labels)

    print(model_name, 'accuracy:', accuracy*100, '%')
    return model
