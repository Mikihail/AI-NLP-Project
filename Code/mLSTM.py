import numpy as np

np.random.seed(1337)  # for reproducibility
import os
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.callbacks import *
from theano import tensor as T
# from visualizer import *
from keras.layers import *
from keras.models import Model

from keras.optimizers import *
from keras.utils.np_utils import to_categorical,accuracy
from keras.layers.core import *


#from keras.utils.visualize_util import plot, to_graph # THIS IS BAD
# from data_reader import *
from reader import *
from myutils import *
import logging
from datetime import datetime
def time_distributed_dense(x, w,
                           input_dim=None, output_dim=None, timesteps=None,repeat_len = None):
    '''Apply y.w + b for every temporal slice y of x.
    '''

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))

    x = K.dot(x, w)
    
    x = K.repeat_elements( x.dimshuffle((0,'x',1)) , repeat_len, axis = 1)
    # reshape to 4D tensor
    x = K.reshape(x, (-1, timesteps,repeat_len,output_dim))
    return x

class mLSTM(Recurrent):
    '''
    Word by Word attention model 

    # Arguments
        output_dim: output_dimensions
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
    # Comments:
        Takes in as input a concatenation of the vectors YH, where Y is the vectors being attended on and
        H are the vectors on which attention is being applied
    # References
        - [REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION](http://arxiv.org/abs/1509.06664v2)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 W_regularizer=None, U_regularizer=None,b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.W1_regularizer = regularizers.get(W_regularizer)
        self.W2_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        supe