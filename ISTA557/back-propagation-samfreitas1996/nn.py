import math
from typing import List

import numpy as np
from scipy.special import expit, logit

# Create a sigmoid function that can be applied to any size of given matrix
def s_matrix(x):
    # input x returns y 
    y = expit(x)
    return y

# create a sigmoid derivative function that can be applied to any size matrix
def sg_matrix(x):
    # create a derivative of the sigmoid function using the inital function and its derivative definition
    y = s_matrix(x)
    z = y*(1-y)
    return z


class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation.
    """

    @classmethod
    def of(cls, *layer_units: int):
        """Creates a single-layer feedforward neural network with the given
        number of units for each layer.

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        pairs = zip(layer_units, layer_units[1:])
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weights matrices, where the
        first is the weights from input units to hidden units, and the last is
        the weights from hidden units to output units.

        :param layer_weights: A list of weight matrices
        """

        # allocate then populate space in the self for the weights
        self.weights = [None]*len(layer_weights)
        i = 0
        for weights in layer_weights:
            # print('\n',weights)
            self.weights[i] = weights
            i += 1


    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """

        # forward propagation

        # set up variables
        a = []
        h = []
        # make the first activation layer the input matrix
        h.append(input_matrix)
        i=0

        # iterate through the weights to perform feed forward
        # where :
        # a - weighted sum
        # h - application of non-linearity, activation layer
        # s_matrix is just a sigmoid funtion that runs on a matrix
        for thisWeight in self.weights:
            a.append(h[i].dot(thisWeight))
            h.append(s_matrix(a[i]))
            i +=1

        # return the last element
        return(h[-1])

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """

        # if the input matrix is above or below 0.5
        # covert the input to 1 or 0
        a = []
        h = []
        h.append(input_matrix)
        i=0
        # iterate through the weights to perform feed forward
        # where :
        # a - activation layer (weighted sum?)
        # h - application of non-linearity 
        # s_matrix is just a sigmoid funtion that runs on a matrix
        for thisWeight in self.weights:
            a.append(h[i].dot(thisWeight))
            h.append(s_matrix(a[i]))
            i +=1

        # from the output of the last layer of the feedforward system
        # use np.where to convert to a binary matrix
        prediction_matrix_binary = np.where(h[-1] < 0.5, 0,1)
        
        return(prediction_matrix_binary)
                        


    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, then applies the following procedure for each input example.
        In the following description, × is matrix multiplication, ⊙ is
        element-wise product, and ⊤ is matrix transpose.

        First, calculate the error, e_L, between last layer's activations, h_L,
        and the output matrix, y. Then calculate g as the element-wise product
        of the error and the sigmoid gradient of last layer's weighted sum
        (before the activation function), a_L.

        e_L = h_L - y
        g = (e_L ⊙ sigmoid'(a_L))⊤

        Then for each layer, l, starting from the last layer and working
        backwards to the first layer, accumulate the gradient for that layer,
        gradient_l, from g and the layer's activations, calculate the error that
        should be backpropagated from that layer, e_l, from g and the layer's
        weights, and calculate g as the element-wise product of e_l and the
        sigmoid gradient of that layer's weighted sum, a_l. Note that h_0 is
        defined to be the input matrix.

        gradient_l += (g × h_l)⊤
        e_l = (weights_l × g)⊤
        g = (e_l ⊙ sigmoid'(a_l))⊤

        When all input examples have applied their updates to the gradients,
        divide each gradient by the number of input examples, N.

        gradient_l /= N

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: same number of matricies as the input weights in the init
        """

        # # forward propagation herdcoded for refrence
        # hi = input_matrix.dot(self.weights[0])
        # h = np.array( [[s(x) for x in row] for row in hi])
        # oi = h.dot(self.weights[1])
        # o = np.array( [[s(x) for x in row] for row in oi])

        # find number of layers
        num_simple_layers = len(self.weights)-1

        # initialize the first layer as the inpur
        # weighted sums
        a = []
        # activation layer
        h = []
        # First activation layer is set to the input matrix
        h.append(input_matrix)

        #forward propagate
        # i is the index 
        i=0
        for thisWeight in self.weights:
            a.append(h[i].dot(thisWeight))
            h.append(s_matrix(a[i]))
            i +=1

        # Back propagation time

        # calc error between last layers activation and input matrix
        # e_L = h_L - y
        e_L = h[-1] - output_matrix
        # calc g as ele-mul of error and sg of last layers weighted sum
        # sg_matrix is a sigmoid gradient fucntion that takes a matrix
        # g = (e_L ⊙ sigmoid'(a_L))⊤
        g = np.multiply(e_L,sg_matrix(a[-1])).T

        # create a blank list with specific size
        gradient_l = [None]*(num_simple_layers+1)
        for i in range(num_simple_layers,-1,-1):

            # gradient_l += (g × h_l)⊤
            gradient_l[i] = np.matmul(g,h[i]).T
            # e_l = (weights_l × g)⊤
            e_l = np.matmul(self.weights[i],g).T
            # g = (e_l ⊙ sigmoid'(a_l))⊤
            g = np.multiply(e_l,sg_matrix(a[i-1])).T

        # create the output gradients to scale with the gradients 
        N = len(input_matrix)
        output_gradient = []
        # print('\nnew grads')
        for grads in gradient_l:
            # print('\n', grads/N)
            output_gradient.append(grads/N)

        return(output_gradient)

    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """

        # initalize the counter 
        iter_coutner = 0

        # loop this interations times 
        while iter_coutner < iterations:
            
            # get the ouput gradients from the gradients function 
            output_gradients = SimpleNetwork.gradients(self,input_matrix,output_matrix)

            # now to update the weights to "train"
            i=0
            for thisWeight in self.weights:
                # iterate through the weights 
                # each updated weight is the inital weight with the product 
                # of the learning rate mutiplied with the output gradents subtracted from it
                updatedWeight = thisWeight - (learning_rate*output_gradients[i])
                self.weights[i] = updatedWeight
                i+=1

            iter_coutner += 1


            