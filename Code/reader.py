# coding: utf-8
import json
from myutils import *
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

def get_data(data,vocab):
    for d in data:
        prem=map_to_idx(tokenize(d["sentence1"]),vocab)
        hyp=map_to_idx(tokenize(d["sentence2"]),vocab)
        label=d["gold_label"]
        yield prem, hyp , label

def load_data(train,vocab,labels={'neutral':0,'entailment':1,'contradiction':2}):
    X,Y,Z=[],[],[]
    for p,h,l in train:
        p=map_to_idx(tokenize(p),vocab)
        h=map_to_idx(tokenize(h),vocab)
        if l in labels:         # get rid of '-'
            X+=[p]
            Y+=[h]
            Z+=[labels[l]]
    return X,Y,Z

def get_vocab(data):
    vocab=Counter()
    for ex in data:
        tokens=tokenize(ex[0])
        tokens+=tokenize(ex[1])
        vocab.update(tokens)
    lst = ["unk", "delimiter"] + [ x for x, y in vocab.iteritems() if y > 0]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    return vocab

def convert2simple(data,out):
    '''
    get the good stuff out of json into a tsv file
    '''
    for d in data:
        print>>out, d["sentence1"]+"\t"+d["sentence2"]+"\t"+d["gold_label"]
    out.close()


if __name__=="__main__":

    train=[l.strip().split('\t') for l in ope