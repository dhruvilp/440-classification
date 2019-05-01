import math
import util
import random
import operator

PRINT = True


class NNClassifier:
    """
    Nearest Neighbors classifier.
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "nearestneighbors"
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()

    def train(self, trainingData, trainingLabels, testData, testLabels, k_number_of_neighbors):

        """YOU CODE HERE"""

    def EuclideanDistance(self, dataOne, dataTwo, length):

        """YOU CODE HERE"""

    def getNeighbors(self, trainingData, trainingLabels, fromTest, k_number_of_neighbors):
        distances = []
        neighbors = []

        """YOU CODE HERE"""
        return neighbors

    def getPredictions(self, neighbors):

        """YOU CODE HERE"""

    def howAccurate(self, testLabels, predictions):

        """YOU CODE HERE"""