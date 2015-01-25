#-----------------------------------------------------------------------------
# Name:        IPUANN - Image Processing Using Artificial Neural Networks
#
# Author:      Walter Sagehorn
#
#-----------------------------------------------------------------------------


import numpy as np
import os

# for getting the data
import gzip
import cPickle

# for generating graphs
import matplotlib
import matplotlib.pyplot as plt


#tanh is the sigmoid function used to determine activation
def sigmoid(x):
    return np.tanh(x)

#the derivative of the sigmoid function, used for backprop
def dsigmoid(x):
    return 1.0 - x**2

def vectorize_tags(data):
    for pair in data:
        temp = pair[1]
        pair[1] = np.zeros(10)
        pair[1][temp] = 1
    return data



class SigmoidLayer():

    """
    desc of sigmoid layer
    """

    def __init__(self, num_in, num_out):

        self.num_in = num_in
        self.num_out = num_out

        #weights initialzed with values of 1
        self.w = numpy.asarray(np.ones(num_in, num_out))

    #x is an array of values
    def activate(self, x):
        return sigmoid(np.dot(self.w, x))

class SoftMaxLayer():
    """
    desc of softmax layer
    """

    def __init__(self, num_in, num_out):

        self.num_in = num_in
        self.num_out = num_out

        #weights initialzed with values of 1
        self.w = numpy.asarray(np.ones(num_in, num_out))

    #x is an array of values
    def activate(self, x):
        return sigmoid(np.dot(self.w, x))


class MLP():

    """
    desc of MLP

    list of hidden layers
    """

    def __init__(self, num_in, hiddenlayers, num_out):
        self.layers = []
        self.layers.append(SigmoidLayer(num_in, hiddenlayers[0]))
        for index in range(len(hiddenlayers) - 2):
            self.layers.append(SigmoidLayer(index, index + 1))
        self.layers.append(SoftMaxLayer(hiddenlayers[-1], num_out))

    #loops through the layers, each one using the previous layer's output as input
    def forward_pass(data):
        for layer in layers:
            data = layer.activate(data)
        return data


    def backprop():
        pass
        #compute derivative of error with respect to weights






def train_MLP(net, num_epochs, validation_interval, validation_size, training_set, validation_set, testing_set):
    epoch = 0
    done = False
    training_error_history = []   # \ __  for graphs
    validation_error_history = [] # /
    lowest_error = np.inf

    while (epoch < num_epochs and not done):
        """
        pseudocode (even though it looks real):

        epoch += 1
        error = net.forward_prop(training_set[0][epoch], training_set[1][epoch])
        training_error_history.append(error)
        net.backprop(error)



    ##### prevent overtraining?

        if epoch % validation_interval == 0:
            total_error = 0
            index = epoch / validation_interval
            for x in range(validation_size):
                total_error += net.forward_prop(validation_set[0][index + x], validation_set[1][index + x])
            average_error = total_error / validation_size
            validation_error_history.append(average_error)

            if average_error > lowest_error:
                done = True
            else:
                lowest_error = average_error
    #####







        """
def test():

    #unzips the data and places it in 3 arrays of 2 dimensions,
        # 1st dimension: an array of arrays holding the values of the pixels on interval [0.0, 1.0)
        # 2nd dimension: an array of the integer corresponding to the pixels
    # for example: training_set[0][5] should yield training_set[1][5]
    print "loading data..."
    pickled_data = gzip.open(os.getcwd() + "/MNIST/mnist.pkl.gz", "rb")
    training_set, validation_set, testing_set = cPickle.load(pickled_data)
    pickled_data.close()
    print "loaded data."

    print "converting tags to vectors..."
    training_set = vectorize_tags(training_set)
    validation_set = vectorize_tags(validation_set)
    testing_set = vectorize_tags(testing_set)
    print "converted tags to vectors."






if __name__ == '__main__':
    test()
