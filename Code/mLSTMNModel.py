import numpy as np

np.random.seed(1337)  # for reproducibility
import os
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.callbacks import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical,accuracy
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from keras.layers import *
#from keras.utils.visualize_util import plot, to_graph # THIS IS BAD
from reader import *
from myutils import *
import logging
from datetime import datetime
from LSTMN import LSTMN

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=300, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=30, dest="xmaxlen", type=int)
    parser.add_argument('-ymaxlen', action="store", default=20, dest="ymaxlen", type=int)
    parser.add_argument('-nopad', action="store", default=False, dest="no_padding", type=bool)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
    parser.add_argument('-l2', action="store", default=0.0003, dest="l2", type=float)
    parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
    parser.add_argument('-local', action="store", default=False, dest="local", type=bool)
    opts = parser.parse_args(sys.argv[1:])
    print "lstm_units", opts.lstm_units
    print "epochs", opts.epochs
    print "batch_size", opts.batch_size
    print "xmaxlen", opts.xmaxlen
    print "ymaxlen", opts.ymaxlen
    print "no_padding", opts.no_padding
    print "regularization factor", opts.l2
    print "dropout", opts.dropout
    return opts

def get_H_n(X):
    ans=X[:, -1, :]  # get last element from time dim
    return ans

def get_H_hypo(X):
    xmaxlen=K.params['xmaxlen']
    return X[:, xmaxlen:, :]  # get elements L+1 to N

def get_WH_Lpi(i):  # get element i
    def get_X_i(X):
        return X[:,i,:];
    return get_X_i

def get_Y(X):
    xmaxlen=K.params['xmaxlen']
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim

def get_R(X):
    Y = X[:,:,:-1]
    alpha = X[:,:,-1]
    tmp=K.permute_dimensions(Y,(0,)+(2,1))  # copied from permute layer, Now Y is (None,k,L) and alpha is always (None,L,1)
    ans=K.T.batched_dot(tmp,alpha)
    return ans

def build_model(opts, verbose=False):

    k = opts.lstm_units
    L = opts.xmaxlen
    N = opts.xmaxlen + opts.ymaxlen + 1  # for delim
    print "x len", L, "total len", N

    input_node = Input(shape=(N,), dtype='int32')

    if opts.local:
        InitWeights = np.load('VocabMat.npy')
    else:   
        InitWeights = np.load('/home/cse/btech/cs1130773/Code/VocabMat.npy')

    emb = Embedding(InitWeights.shape[0],InitWeights.shape[1],input_length=N,weights=[InitWeights])(input_node)
    d_emb = Dropout(0.1)(emb)

    premise = Lambda(get_Y,output_shape=(L, 300))(d_emb)
    hypothesis = Lambda(get_H_hypo, output_shape=(N-L, 300))(d_emb)

    Y = LSTMN(opts.lstm_units,return_sequences=True)(premise)

    h_hypo = LSTMN(opts.lstm_units,return_sequences=True)(hypothesis)

    WY = TimeDistributed(Dense(k,W_regularizer=l2(0.01)))(Y)
    Wh_hypo = TimeDistributed(Dense(k,W_regularizer=l2(0.01)))(h_hypo)

    # GET R1
    f = get_WH_Lpi(0)
    Wh_lp = [Lambda(f, output_shape=(k,))(Wh_hypo)]
    Wh_lp_cross_e = [RepeatVector(L)(Wh_lp[0])]

    Sum_Wh_lp_cross_e_WY = [merge([Wh_lp_cross_e[0], WY],mode='sum')]
    M = [Activation('tanh')(Sum_Wh_lp_cross_e_WY[0])]    

#    alpha_TimeDistributedDense_Layer = TimeDistributed(Dense(1,activation='softmax'))
    Distributed_Dense_init_weight = ((2.0/np.sqrt(k)) * np.random.rand(k,1)) - (1.0 / np.sqrt(k))
    Distributed_Dense_init_bias = ((2.0) * np.random.rand(1,)) - (1.0)
    alpha = [TimeDistributed(Dense(1,activation='softmax', weights