__author__ = 'blaise'

from utils import constants
from customIO.load import libsvm
from configs import configSpark
from models.spark import RF


config = configSpark.ConfigSpark()

loader = libsvm.Loader(config.sc,constants.LOCAL_DATA_PATH)

(trainingData, testData) = loader.data.randomSplit([0.7, 0.3])


learner = RF.SparkRF(config.sc)

learner.evaluate(trainingData,testData)