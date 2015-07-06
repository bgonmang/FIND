__author__ = 'blaise'

from models import BaseLearner
from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest

from utils.constants import *

class SparkRF:

    def __init__(self,sc):
        #sc = SparkContext(SPARK_SERVER_NAME, SPARK_APP_NAME)
        self.model = None

    def train(self, trainingData):
        self.model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=10, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

    def predict(self, testData):
        predictions = self.model.predict(testData.map(lambda x: x.features))
        return predictions

    def evaluate(self, trainingData,  testData=None, metric=None):
        if testData !=None:
            model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=10, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)
            predictions = model.predict(testData.map(lambda x: x.features))
            labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
            testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
            print('Test Error = ' + str(testErr))
        else: #cross validation
            pass