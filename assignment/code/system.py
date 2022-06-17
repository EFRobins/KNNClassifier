"""
Finished project
By Eric Robins
"""

from scipy import linalg as LA
from collections import Counter
import json
from typing import Collection, List
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
#used to help debug 

#feature selection runs to make this 10
N_DIMENSIONS = 40

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
       
 """

    #compact implementation of knn adapted from lab 6
    x = np.dot(test, train.transpose())
    tst = np.sqrt(np.sum(test * test, axis=1))
    trn = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(tst, trn.T)
    neighbours = np.argsort(-dist, axis=1)

    #finding the closest 6 neighbours and the most common label
    #used 6 after running and finding the most accurate number
    l = []
    for char in range(neighbours.shape[0]):
        nearest_neighbours = Counter(train_labels[neighbours[char][:6]]).most_common(1)
        nearest = np.array(nearest_neighbours)[0][0]
        l.append(nearest)  
    return np.asarray(l)


#calculates the euclidean distance of one test value to the entierity of the training data
#returns a list of indexes of the closest neighbours
#unused implementation of KNN as it involves loops and is slow
def getClosestNeighbours(train: np.ndarray, test: np.ndarray) -> List[float]:
    k_closest = [] #stores indexes of lowest values
    val = 0 #stores current index
    
    for row in train:
        dist = np.linalg.norm(test-row)
        k_closest.append([val,dist])
        val += 1
    k_closest = sorted(k_closest, key=lambda x: x[1], reverse=True)
    k_closest = k_closest[:10]
    return k_closest

#makes use of np.gradient to calculate the divergence for feature selection
def measureDivergence(feature1: np.ndarray, feature2: np.ndarray) -> float:
    return np.ufunc.reduce(np.add,np.gradient(np.stack((feature1, feature2))))

#narrows down 40 features to 10 based on highest divergence
def feature_select(features: np.array)-> np.array: 
    features = features.T
    divergences = []
    for i in features:
        list = []
        for y in features:
            x = measureDivergence(i,y)
            list.append(x)
        divergences.append([sum(list),i])
    divergences = sorted(divergences,key=lambda x: [0], reverse=True)
    divergences = divergences[:10]
    best_features = []
    for i in divergences:
        best_features.append(i[1])
    best_features = np.asarray(best_features)
    return best_features.T


#applies same pca transformation to training data and test data based on transform for the training data
def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    data -= model["mean"]
    evecs = np.array(model["evectors"])
    reduced_data = np.dot(evecs.T,data.T).T
    return feature_select(reduced_data)


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model["labels_train"] = labels_train.tolist()
    model["mean"] = fvectors_train.mean(axis=0).tolist()
    X = np.cov(fvectors_train,rowvar=False)
    evals, evecs = LA.eigh(X)
    index = np.argsort(evals)[::-1]
    evecs = evecs[:,index]
    evecs = evecs[:, :N_DIMENSIONS]
    
    evals = evals[index]
    model["evectors"] = evecs.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """

    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)

