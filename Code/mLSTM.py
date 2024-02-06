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
       