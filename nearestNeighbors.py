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

        test = []
        predictions = []
        for x in range(len(testData)):
            part = list(testData[x].values())
            test.append(part)

        for x in range(len(test)):
            neighbors = self.getNeighbors(trainingData, trainingLabels, test[x], k_number_of_neighbors)
            predict = self.getPredictions(neighbors)
            predictions.append(predict)
            accurate = self.howAccurate(testLabels, predictions)
            # print("Predicted: " + repr(predict) + " Actual: " + repr(testLabels[x]))

        print("Accuracy: " + repr(accurate) + "%")

    def EuclideanDistance(self, dataOne, dataTwo, length):
        distance = 0
        for x in range(length):
            distance += pow((dataOne[x] - dataTwo[x]), 2)
        return math.sqrt(distance)

    def getNeighbors(self, trainingData, trainingLabels, fromTest, k_number_of_neighbors):
        distances = []
        neighbors = []
        neighborLabels = []
        train = []
        for x in range(len(trainingData)):
            part = list(trainingData[x].values())
            train.append(part)

        length = len(fromTest) - 1
        for x in range(len(trainingData)):
            dist = self.EuclideanDistance(fromTest, train[x], length)
            distances.append((trainingData[x], dist, trainingLabels[x]))
        distances.sort(key=lambda tup: tup[1])

        for x in range(k_number_of_neighbors):
            neighbors.append(distances[x])

        return neighbors

    def getPredictions(self, neighbors):
        labels = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in labels:
                labels[response] += 1
            else:
                labels[response] = 1

        sortedLabels = sorted(labels.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedLabels[0][0]

    def howAccurate(self, testLabels, predictions):
        correct = 0
        for x in range(len(predictions)):
            if testLabels[x] == predictions[x]:
                correct += 1

        return (correct / float(len(testLabels))) * 100