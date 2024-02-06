
import numpy as np
import sys
from myutils import *
from process import *

DIM_SIZE = 300
WINDOW = 4

def data2vec(data,labels,RMat,dictionary):
	flag = 0
	X_train = []
	label = []
	for p,h,l in data:
		if(l not in labels):
			continue
		label.append(labels[l])