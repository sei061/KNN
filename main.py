import math

import pandas as pd
from math import sqrt

# Iris dataset link
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df_iris = pd.read_csv(URL, header=None, names=['sepal length','sepal width','petal length','petal width','class'])
df_iris.loc[df_iris['class'] == 'Iris-setosa', df_iris.columns == 'class'] = 0
df_iris.loc[df_iris['class'] == 'Iris-versicolor', df_iris.columns == 'class'] = 1
df_iris.loc[df_iris['class'] == 'Iris-virginica', df_iris.columns == 'class'] = 2


codes, uniques = pd.factorize(df_iris['class'])

factorized = pd.array(df_iris, bytearray)


new_dp = [7.0, 3.1, 1.3, 0.7]




def euclidean_distance(p, q):
    distance = []
    for i in range(len(q)):
        distance.append(math.pow(p[i] - q[i], 2))
        dist = sum(distance)
        return sqrt(dist)

"""

    The euclidean_distance() function takes in two lists as its arguments.
    The distance list is initialized to an empty list.
    The for loop iterates over the length of the q list.
    The distance list is appended with the squared difference between the p and q values.
    The sum() function is called on the distance list to sum all the values in the list.
    The sqrt() function is called on the sum of the values in the distance list to compute the square root of the sum.
    The euclidean_distance() function returns the square root of the sum.
    
    p is the first row of data, q is the second row of data, i is the index to a specific column as we sum across all
    columns.

"""


def get_neighbors(train, test_row, k):
    distances = []
    for train_row in train:
        dist = (train_row, euclidean_distance(test_row, train_row))
        distances.append(dist)
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

"""
    Calculate the distance between the test point and the current point from the training set.
    Add the distance and the index of the point in a tuple.
    Add tuple to a list containing all distances for all points in the training set.
    Sort the list of distances in ascending order.
    Get the top k elements from the sorted list.
    Get the labels of the selected k elements.
    Return the labels as the prediction for the test point.
    
    Once the distances are calculated we sort all of the records in the training set by their distance to the new data.
    top k returns as the most similar neighbor. distance of each record as a tuple, sort the list of tuples by the
    distance and then retrieve the neighbors.
    
    

"""


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = []
    for row in neighbors:
        output_values.append(row[len(row) -1])
    prediction = max(output_values, key=output_values.count)
    return prediction
"""
    Get the nearest neighbors.
    Get the output values for those neighbors.
    Return the most common class.
    
    Achieved by performing the max() function on the list of output values.
"""


predictions = predict_classification(factorized, new_dp, 5)

print('Classified data: %s, Most common class value %s.' % (new_dp, uniques[predictions]))








