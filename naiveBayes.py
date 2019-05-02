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

        global highestPP, highestCP
        "*** YOUR CODE HERE ***"
        # Define Prior Probabilities:
        """
        Estimating P(Y) directly from the training data: P(Y) = c(y)/n
        Where c(y) = # of training instances with label y and n = total # of training instances.
        """
        prior_prob = util.Counter()

        """
        y: P(F_i | Y = y)
        P(F_i = f_i | Y = y) = c(f_i, y) / Sum(f_i \in {0,1})(f'_i, y)
        where c(f_i, y) = # of times pixel F_i took value f_i in the training examples of label y.
        """
        # Define conditional probabilities:
        cond_prob = util.Counter()

        # Define frequencies for conditional probabilities:
        freq_cond_prob = util.Counter()  # num of time features seen

        # Calculate prior probabilities:
        for label in trainingLabels:
            prior_prob[label] = prior_prob[label] + 1

        # Normalize:
        prior_prob.normalize()

        # To find conditional probabilities (Training):
        for i in range(0, len(trainingData)):
            datum = trainingData[i]
            label = trainingLabels[i]
            for pixel, value in datum.items():
                freq_cond_prob[(pixel, label)] = freq_cond_prob[(pixel, label)] + 1
                if value == 1:
                    cond_prob[(pixel, label)] = cond_prob[(pixel, label)] + 1

        highest = 0

        # Tuning with LaPlace Smoothing (Tuning)
        for k in kgrid:
            # Keep track of conditional probabilities and frequencies
            k_cond_prob = util.Counter()
            k_freq = util.Counter()

            # Plugging in for a new computation from previous values
            for pixelANDlabel, value in cond_prob.items():
                k_cond_prob[pixelANDlabel] = value

            for pixelANDlabel, value in freq_cond_prob.items():
                k_freq[pixelANDlabel] = value

            # Adding k value
            for pixel in self.features:
                for label in self.legalLabels:
                    k_cond_prob[(pixel, label)] = k_cond_prob[(pixel, label)] + k
                    k_freq[(pixel, label)] = k_freq[(pixel, label)] + k

            for i, j in k_cond_prob.items():
                k_cond_prob[i] = float(j) / k_freq[i]

            self.CP = k_cond_prob.copy()
            self.PP = prior_prob.copy()

            # Starting the Classification:
            correct_predictions = 0
            predictions = self.classify(validationData)

            for i in range(0, len(validationData)):
                if validationLabels[i] == predictions[i]:
                    correct_predictions = correct_predictions + 1

            # evaluating performance on validation set
            percentage = (float(correct_predictions) / len(validationLabels)) * 100

            print("k = ", k, " accuracy is: ", percentage, "%")

            if percentage > highest:
                highest = percentage
                highestPP = self.PP.copy()
                highestCP = self.CP.copy()

        self.PP = highestPP.copy()
        self.CP = highestCP.copy()

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
        for i in self.legalLabels:
            logJoint[i] = math.log(self.PP[i])
            for pixel, value in datum.items():
                if value == 0:
                    probability = 1 - self.CP[(pixel, i)]
                    logJoint[i] = logJoint[i] + math.log(probability if probability > 0 else 1)
                else:
                    "if value != 0 --> add"
                    logJoint[i] = logJoint[i] + math.log(self.CP[(pixel, i)])

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
