__author__ = 'blaise'
from pyspark.mllib.util import MLUtils

class Loader:
    def __init__(self,sc, path):
        # Load and parse the data file into an RDD of LabeledPoint.
        self.data = MLUtils.loadLibSVMFile(sc, path)

