#-----------------------------------------------------------------------------
# Name:        IPUANN - Image Processing Using Artificial Neural Networks
#
# Author:      Walter Sagehorn
#
#-----------------------------------------------------------------------------


import numpy as np
import os, time

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

#softmax function
def softmax(x):
    e = np.exp(np.array(x))
    total = e / np.sum(e)
    return total

#converts each integer a one-hot vector with the said integer hot
def vectorize_tags(data):
    tags = [None] * len(data[1])
    retdata = [data[0], None]
    for i, num in enumerate(data[1]):
        temp = num
        tags[i] = np.zeros(10)
        tags[i][temp] = 1
    retdata[1] = list(tags)
    return retdata


class Layer(object):
    def __init__(self, num_in, num_out):

        self.num_in = num_in
        self.num_out = num_out

        # weights initialzed with values from sqrt(-6./(n_in+n_hidden)) to sqrt(6./(n_in+n_hidden))
        self.w = np.asarray(
            np.random.RandomState(1234).uniform(
                low=-np.sqrt(6. / (num_in + num_out)),
                high=np.sqrt(6. / (num_in + num_out)),
                size=(num_in, num_out)
            )
        )

    # x is an array of values
    def activate(self, x):
        return 0


class SigmoidLayer(Layer):
    """
    desc of sigmoid layer
    """

    def __init__(self, num_in, num_out):
        Layer.__init__(self, num_in, num_out)

    def activate(self, x):
        return sigmoid(np.dot(x, self.w))

class SoftMaxLayer(Layer):
    """
    desc of softmax layer
    """

    def __init__(self, num_in, num_out):
        Layer.__init__(self, num_in, num_out)

    def activate(self, x):
        return softmax(np.dot(x, self.w))


class MLP(object):

    """
    desc of MLP

    list of hidden layers
    """

    def __init__(self, num_in, hiddenlayers, num_out):
        self.layers = []
        self.layers.append(SigmoidLayer(num_in, hiddenlayers[0]))
        for index in range(len(hiddenlayers) - 1):
            self.layers.append(SigmoidLayer(hiddenlayers[index], hiddenlayers[index + 1]))
        self.layers.append(SoftMaxLayer(hiddenlayers[-1], num_out))

    #loops through the layers, each one using the previous layer's output as input
    def forward_pass(self, data):
        for layer in self.layers:
            data = layer.activate(data)
        return data


    def backprop():
        pass
        #compute derivative of error with respect to weights






def train_MLP(net, num_epochs, validation_interval, validation_size, training_set, validation_set):
    epoch = 0
    done = False
    training_error_history = []   # \ __  for graphs
    validation_error_history = [] # /
    lowest_error = np.inf
    error = 0

    start_time = time.clock()
    while (epoch < num_epochs and not done):
        epoch += 1
        print "training epoch {0}/{1}.".format(epoch, num_epochs)
        net_output = net.forward_pass(training_set[0][epoch])
        print net_output
        training_error_history.append(error)
        #net.backprop(error)

    end_time = time.clock()
    print "done training. Time elapsed: {0}m.".format(((end_time - start_time) / 60.))

def test_MLP(net, num_epochs, training_set):
    epoch = 0
    done = False
    testing_error_history = []
    num_correct = 0

    start_time = time.clock()
    while (epoch < num_epochs and not done):
        epoch += 1
        print "testing epoch {0}/{1}.".format(epoch, num_epochs)
        net_output = net.forward_pass(training_set[0][epoch])
        #testing_error_history.append(error)

    end_time = time.clock()
    print "done testing"
    print "correct classifications: {0} out of {1}.".format(num_correct, num_epochs)
    print "percent correct: {0}".format( 1. * num_correct / num_epochs)
    print "Time elapsed: {0}m.".format(((end_time - start_time) / 60.))


"""
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

def load_data():
    pickled_data = gzip.open(os.getcwd() + "/MNIST/mnist.pkl.gz", "rb")
    training_set, validation_set, testing_set = cPickle.load(pickled_data)
    pickled_data.close()
    return training_set

def test():

    # unzips the data and places it in 3 arrays of 2 dimensions,
        # 1st dimension: an array of arrays holding the values of the pixels on interval [0.0, 1.0)
        # 2nd dimension: an array of the integers corresponding to the pixels
    # for example: training_set[0][5] should yield training_set[1][5]
    print "loading data..."
    pickled_data = gzip.open(os.getcwd() + "/MNIST/mnist.pkl.gz", "rb")
    training_set, validation_set, testing_set = cPickle.load(pickled_data)
    pickled_data.close()
    print "loaded data."

    # converts set[1] into one-hot vectors so that they can be more
    # easily compared to the network output.
    # for example: vectorize(3) -> [0,0,0,1,0,0,0,0,0,0]
    print "converting tags to vectors..."
    training_set = vectorize_tags(training_set)
    validation_set = vectorize_tags(validation_set)
    testing_set = vectorize_tags(testing_set)
    print "converted tags to vectors."

    hiddenlayers = [2500,2000,1500,1000,500]
    network = MLP(784, hiddenlayers, 10)
    print "created network."

    print "training network..."
    train_MLP(network, 10, 5, 2, training_set, validation_set)

    print "testing network..."
    test_MLP(network, 10, training_set)






if __name__ == '__main__':
    test()
