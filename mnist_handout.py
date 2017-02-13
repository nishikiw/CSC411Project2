from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
import os

import cPickle

import os
from scipy.io import loadmat


def main():

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")
    
    # divide all data point by 255.0
    
    for digit in range(0,10):
        train = "train" + str(digit)
        M[train] = M[train].astype(float)
        for i in range(0, len(M[train])):
            M[train][i] = M[train][i]/255.0
            # maybe should reshape to (784,1). Currently it's (784, )
    
    # Display 10 images of each.
    np.random.seed(0)
    gs = GridSpec(10, 10)
    for digit in range(0, 10):
        train = "train" + str(digit)
        set = [int(i) for i in np.random.sample(10)*len(M[train])]
        for i in range(0, 10):
            ax = plt.subplot(gs[10*digit+i])
            ax.imshow(M[train][set[i]].reshape((28, 28)), cmap = cm.gray)
    
    # Save the figure for part1 if it's not already in current folder
    if not os.path.exists("part1.png"):
        plt.savefig('part1.png')


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)


def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output


# p is calculated results and y is actual results
# this works or vector
def NLL(p, y):
    return -sum(y*log(p)) 
    
    
def NLL_gradient(w, y, x):
    p = lin_combin(w, b, x)
    return dot(x, (p - y).T)
    
    
def grad_descent(f, df, x, y, init_w, init_b, alpha, max_iteration):
    EPS = 1e-10   #EPS = 10**(-10)
    prev_w = init_w-10*EPS
    w = init_w.copy()
    iter  = 0
    while norm(w - prev_w) >  EPS and iter < max_iter:
        prev_w = w.copy()
        p = lin_combin(w, b, x)
        w -= alpha*df(p, y, x)
        iter += 1
        if (iter % 1000 == 0):
            print("iter", iter, "NLL(x)", f(p, y))
    return w


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    
    
# follow the diagram
# for part 2, W should be 784 x 10. b should be 10 x 1
def lin_combin(w, b, x):
    o = (dot(w.T, x) + b)
    return softmax(o)


# x should be 784 x 5000
# y should be 10 x 5000
# b should be same as y
# w should be 784 x 10




'''
#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"] #784 x 300
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"] #300 x 10
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
'''