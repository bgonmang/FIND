__author__ = 'blaise'

class BaseLearner:

    def __init__(self):
        pass

    def train(self, traindata):
        pass

    def predict(self, testdata):
        pass

    def evaluate(self, traindata, nfold=5):
        pass