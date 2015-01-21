#-----------------------------------------------------------------------------
# Name:        IPUANN - Image Processing Using Artificial Neural Networks
#
# Author:      Walter Sagehorn
#
#-----------------------------------------------------------------------------


"""
imports



helper methods (sigmoid)

class hidden layer



class MLP
nin
hidden array of ints [2500,2000,1500,1000,500]
nout


def test_mlp(rate, epochs, validation_interval):



"""


import numpy as np
import os


import gzip
import cPickle

class HiddenLayer():

    """
    desc of hidden layer
    """

    def __init__(self, data_in, num_in, num_nodes):
        pass





class MLP():

    """
    desc of MLP
    """

    def __init__(self, hiddenlayers):
        pass











def test():

    #unzips the data and places it in 3 arrays of 2 dimensions,
        # 1st dimension: an array of arrays holding the values of the pixels
        # 2nd dimension: an array of the integer corresponding to the pixels
    # for example: training_set[0][5] should evaluate to training_set[1][5]
    pickled_data = gzip.open(os.getcwd() + "/MNIST/mnist.pkl.gz", "rb")
    training_set, validation_set, testing_set = cPickle.load(pickled_data)
    pickled_data.close()
    print "loaded data"




if __name__ == '__main__':
    test()
