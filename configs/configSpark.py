__author__ = 'blaise'

from utils.constants import *
from configs import Config
from pyspark import SparkContext

class ConfigSpark:
    def __init__(self):
        self.sc = SparkContext(SPARK_SERVER_NAME, SPARK_APP_NAME)
