
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
    parser.add_argument('-lstm', action="store", default=150, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=30, dest="xmaxlen", type=int)
    parser.add_argument('-ymaxlen', action="store", default=20, dest="ymaxlen", type=int)
    parser.add_argument('-nopad', action="store", default=False, dest="no_padding", type=bool)
    parser.add_argument('-lr', action="store", default=0.0008, dest="lr", type=float)
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

    k = 2 * opts.lstm_units
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
    forward = LSTMN(opts.lstm_units,return_sequences=True)(d_emb)
    backward = LSTMN(opts.lstm_units,return_sequences=True,go_backwards=True)(d_emb)
    forward_backward = merge([forward,backward],mode='concat',concat_axis=2)
    dropout = Dropout(0.1)(forward_backward)

    h_n = Lambda(get_H_n,output_shape=(k,))(dropout)

    Y = Lambda(get_Y,output_shape=(L, k))(dropout)
    WY = TimeDistributed(Dense(k,W_regularizer=l2(0.01)))(Y)

    ###########
    # "dropout" layer contains all h vectors from h_1 to h_N

    h_hypo = Lambda(get_H_hypo, output_shape=(N-L, k))(dropout)
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
    alpha = [TimeDistributed(Dense(1,activation='softmax', weights=[Distributed_Dense_init_weight, Distributed_Dense_init_bias]), name='alpha1')(M[0])]

    Join_Y_alpha = [merge([Y, alpha[0]],mode='concat',concat_axis=2)]    
    _r = [Lambda(get_R, output_shape=(k,1))(Join_Y_alpha[0])]
    r = [Reshape((k,))(_r[0])]

#    Tan_Wr_Dense_Layer = Dense(k,W_regularizer=l2(0.01),activation='tanh')
#    Wr_Dense_layer = Dense(k,W_regularizer=l2(0.01))
    Tan_Wr_init_weight = 2*(1/np.sqrt(k))*np.random.rand(k,k) - (1/np.sqrt(k))
    Tan_Wr_init_bias = 2*(1/np.sqrt(k))*np.random.rand(k,) - (1/np.sqrt(k))
    Tan_Wr = [Dense(k,W_regularizer=l2(0.01),activation='tanh', name='Tan_Wr1', weights=[Tan_Wr_init_weight, Tan_Wr_init_bias])(r[0])]

    Wr_init_weight = 2*(1/np.sqrt(k))*np.random.rand(k,k) - (1/np.sqrt(k))
    Wr_init_bias = 2*(1/np.sqrt(k))*np.random.rand(k,) - (1/np.sqrt(k))
    Wr = [Dense(k,W_regularizer=l2(0.01), name='Wr1', weights=[Wr_init_weight, Wr_init_bias])(r[0])]
    Wr_cross_e = [RepeatVector(L)(Wr[0])]

    star_r = []
    # GET R2, R3, .. R_N
    for i in range(2,N-L+1):
        f = get_WH_Lpi(i-1)
        Wh_lp.append( Lambda(f, output_shape=(k,))(Wh_hypo) )
        Wh_lp_cross_e.append( RepeatVector(L)(Wh_lp[i-1]) )

        Sum_Wh_lp_cross_e_WY.append( merge([Wh_lp_cross_e[i-1], WY, Wr_cross_e[i-2]],mode='sum') )
        M.append( Activation('tanh')(  Sum_Wh_lp_cross_e_WY[i-1] ) )
        alpha.append( TimeDistributed(Dense(1,activation='softmax'), name='alpha'+str(i))(M[i-1]) )

        Join_Y_alpha.append( merge([Y, alpha[i-1]],mode='concat',concat_axis=2) )
        _r.append( Lambda(get_R, output_shape=(k,1))(Join_Y_alpha[i-1]) )
        star_r.append( Reshape((k,))(_r[i-1]) )
        r.append( merge([star_r[i-2], Tan_Wr[i-2]], mode='sum') )

        if i != (N-L):
            Tan_Wr.append( Dense(k,W_regularizer=l2(0.01),activation='tanh', name='Tan_Wr'+str(i))(r[i-1]) )
            Wr.append( Dense(k,W_regularizer=l2(0.01), name='Wr'+str(i))(r[i-1]) )
            Wr_cross_e.append( RepeatVector(L)(Wr[i-1]) )


    Wr = Dense(k,W_regularizer=l2(0.01))(r[N-L-1]) 
    Wh = Dense(k,W_regularizer=l2(0.01))(h_n)
    Sum_Wr_Wh = merge([Wr, Wh],mode='sum')
    h_star = Activation('tanh')(Sum_Wr_Wh)    

    out = Dense(3, activation='softmax')(h_star)
    model = Model(input = input_node ,output = out)
    model.summary()

#        graph = to_graph(model, show_shape=True)
#        graph.write_png("model2.png")

    model.compile(loss='categorical_crossentropy', optimizer=Adam(options.lr), metrics=['accuracy'])
    return model


def compute_acc(X, Y, vocab, model, opts, filename=None):
    scores=model.predict(X,batch_size=options.batch_size)
    prediction=np.zeros(scores.shape)
    for i in range(scores.shape[0]):
        l=np.argmax(scores[i])
        prediction[i][l]=1.0
    assert np.array_equal(np.ones(prediction.shape[0]),np.sum(prediction,axis=1))
    plabels=np.argmax(prediction,axis=1)
    tlabels=np.argmax(Y,axis=1)
    acc = accuracy(tlabels,plabels)

    if filename!=None:
        f = open(filename,'w')
        for i in range(len(X)):
            f.write(map_to_txt(X[i],vocab)+ " : "+ str(plabels[i])+ "\n")
        f.close()

    return acc

def getConfig(opts):
    conf=[opts.xmaxlen,
          opts.ymaxlen,
          opts.batch_size,
          opts.lr,
          opts.lstm_units,
          opts.epochs]
    if opts.no_padding:
        conf.append("no-pad")
    return "_".join(map(lambda x: str(x), conf))

def save_model(model,wtpath,archpath,mode='yaml'):
    if mode=='yaml':
        yaml_string = model.to_yaml()
        open(archpath, 'w').write(yaml_string)
    else:
        with open(archpath, 'w') as f:
            f.write(model.to_json())
    model.save_weights(wtpath)


def load_model(wtpath,archpath,mode='yaml'):
    if mode=='yaml':
        model = model_from_yaml(open(archpath).read())
    else:
        with open(archpath) as f:
            model = model_from_json(f.read())
    model.load_weights(wtpath)
    return model


def concat_in_out(X,Y,vocab):
    numex = X.shape[0] # num examples
    glue=vocab["delimiter"]*np.ones(numex).reshape(numex,1)
    inp_train = np.concatenate((X,glue,Y),axis=1)
    return inp_train

class WeightSharing(Callback):
    def __init__(self, shared):
        self.shared = shared