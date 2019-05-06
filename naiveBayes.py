# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.extra = False

    def setSmoothing(self, k):
        """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
    Outside shell to call your method. Do not modify this method.
    """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in trainingData for f in datum.keys()]));

        if self.automaticTuning:
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

        priorDist = util.Counter()

        condProb = {0: util.Counter(), 1: util.Counter()}
        # condProb = util.Counter() # conditional probability of features (feature, value):(count)
        count = util.Counter()  # number of time feature seen (feature, value):(count)

        # Training
        for i in range(len(trainingData)):
            datum = trainingData[i]
            label = trainingLabels[i]
            priorDist[label] += 1
            for feat, val in datum.items():
                count[(feat, label)] += 1
                if val > 0:
                    condProb[1][(feat, label)] += 1
                else:
                    condProb[0][(feat, label)] += 1

        priorDist.normalize()
        self.priorDist = priorDist

        bestAccuracy = -1
        for k in kgrid:
            # smoothing:
            smoothCondProb = {0: condProb[0].copy(), 1: condProb[1].copy()}
            smoothCount = count.copy()

            for label in self.legalLabels:
                for feat in self.features:
                    smoothCondProb[0][(feat, label)] += k
                    smoothCondProb[1][(feat, label)] += k
                    smoothCount[(feat, label)] += 2 * k

            # normalizing:
            for fkey, cnt in smoothCondProb[0].items():
                smoothCondProb[0][fkey] = (cnt * 1.0) / smoothCount[fkey]
            for fkey, cnt in smoothCondProb[1].items():
                smoothCondProb[1][fkey] = (cnt * 1.0) / smoothCount[fkey]

            self.condProb = smoothCondProb

            if self.extra:
                # evaluating performance on validation set
                guess = self.classify(validationData)
                accuracyCount = [guess[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                print("k=", k, " accuracy:", (100.0 * accuracyCount / len(validationLabels)))

                if accuracyCount > bestAccuracy:
                    bestParam = (smoothCondProb, k)
                    bestAccuracy = accuracyCount

        if self.extra:
            self.condProb, self.k = bestParam

    def classify(self, testData):
        """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        for label in self.legalLabels:
            logJoint[label] = math.log(self.priorDist[label])
            for feature, value in datum.items():
                try:
                    if value > 0:
                        logJoint[label] += math.log(self.condProb[1][feature, label])
                        logJoint[label] += math.log(1 - self.condProb[0][feature, label])
                    else:
                        logJoint[label] += math.log(self.condProb[0][feature, label])
                        logJoint[label] += math.log(1 - self.condProb[1][feature, label])
                except:
                    print(self.condProb[1][feature, label])
                    print(self.condProb[0][feature, label])
                    util.raiseNotDefined()

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2)
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        # Not needed so not implemented
        util.raiseNotDefined()

        return featuresOdds
