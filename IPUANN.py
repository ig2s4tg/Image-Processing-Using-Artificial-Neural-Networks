#-----------------------------------------------------------------------------
# Name:        IPUANN - Image Processing Using Artificial Neural Networks
#
# Author:      Walter Sagehorn
#
#-----------------------------------------------------------------------------


import numpy as np
import os


import gzip
import cPickle


#tanh is the sigmoid function used to determine activation
def sigmoid(x):
    return np.tanh(x)

#the derivative of the sigmoid function, used for backprop
def dsigmoid(x):
    return 1.0 - x**2



class HiddenLayer():

    """
    desc of hidden layer
    """

    def __init__(self, num_in, num_out):

        self.num_in = num_in
        self.num_out = num_out
        self.bias= 0 #for now

        #weights initialzed with values of 1
        self.w = numpy.asarray(np.ones(num_in, num_out))

    #x is an array of values
    def activate(self, x):
        return (x * self.w) + self.bias




class MLP():

    """
    desc of MLP
    """

    def __init__(self, num_in, hiddenlayers, num_out):
        pass

    def activate():
        pass






def train_MLP(net, num_epochs, validation_interval, validation_size, training_set, validation_set, testing_set):
    epoch = 0
    done = False
    lowest_error = np.inf

    while (epoch < num_epochs and not done):
        """
        pseudocode (even though it looks real):

        epoch += 1
        error = net.forward_prop(training_set[0][epoch], training_set[1][epoch])
        net.backprop(error)



    ##### prevent overtraining?

        if epoch % validation_interval == 0:
            total_error = 0
            index = epoch / validation_interval
            for x in range(validation_size):
                total_error += net.forward_prop(validation_set[0][index + x], validation_set[1][index + x])
            average_error = total_error / validation_size

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
    pickled_data = gzip.open(os.getcwd() + "/MNIST/mnist.pkl.gz", "rb")
    training_set, validation_set, testing_set = cPickle.load(pickled_data)
    pickled_data.close()
    print "loaded data"




if __name__ == '__main__':
    test()
