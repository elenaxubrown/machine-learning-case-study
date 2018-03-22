import numpy as np
import operator

def euclidean_distance(input1, input2):
    """
    Compute the euclidean distance between input1 and input2

    :param input1: input data point 1, a Python list
    :param input2: input data point 2, a Python list
    :return: euclidean distance between input1 and input2, a float number
    """

    input1 = np.array(input1)
    input2 = np.array(input2)
    d = np.sqrt(np.sum((input1-input2)**2))
    return d

def get_neighbors_indices(training_inputs, test_instance, k):
    """
    Get the indices of k closest neighbors to testInstance using Euclidean Distance
    Use the euclidean_distance helper function

    :param training_inputs: inputs of training data, a 2D Python list
    :param test_instance: an instance input of test data, a Python list
    :param k: number of neighbors used, an int
    """

    dist = []
    for i in range(len(training_inputs)):
        distance = euclidean_distance(test_instance, training_inputs[i])
        dist.append((i,distance))
    dist.sort(key = operator.itemgetter(1))
    indice = []
    for j in range(k):
        indice.append(dist[j][0])
    return indice
def get_response(training_labels, neighbor_indices):
    """
    Get the most commonly voted response from a number of neighbors

    :param training_labels: labels of training data, a Python list
    :param neighbor_indices: a Python list of indices of k closest neighbors from the training data
    :return: the class/label with the highest vote, an int
    """

    vote = {}
    for i in neighbor_indices:    
        if training_labels[i] in vote:
            vote[training_labels[i]] += 1
        else:
            vote[training_labels[i]] = 1
    sortvote = sorted(vote.items(), key = operator.itemgetter(1),reverse = True)
    return sortvote[0][0]
