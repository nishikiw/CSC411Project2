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
import math

import cPickle

import os
from scipy.io import loadmat


def main():

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")
    
    # divide all data point by 255.0
    for set in ["train", "test"]:
        for digit in range(0,10):
            set_name = set + str(digit)
            M[set_name] = M[set_name].astype(float)
            for i in range(0, len(M[set_name])):
                M[set_name][i] = M[set_name][i]/255.0
    
    np.random.seed(0)
    #part1(M)
    #part3b(M)
    part4(M)
    

def part1(M):
    """Coding part for part 1."""
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


def part3b(M):
    """Test gradient for part 3b."""
    
    x1 = M["train8"][0]
    x2 = M["train8"][1]
    x = vstack((x1, x2)).T
    y = array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    w = zeros((x.shape[0], 10))
    b = zeros((10, 2))
    test_gradient(x, y, b, w)


def part4(M):
    """Coding part for part 4."""
    
    # set up training set
    if not os.path.exists("x.txt"):
        x = setup_x(M, "train")
        np.savetxt("x.txt", x)
    else:
        x = loadtxt("x.txt");
    print("x is set up")
    
    if not os.path.exists("y.txt"):
        y = setup_y(M, "train")
        np.savetxt("y.txt", y)
    else:
        y = loadtxt("y.txt")
    print("y is set up")
    training_size = int(y.shape[1]) # there are 60000 sample in training set
    
    # set up test set
    if not os.path.exists("x_test.txt"):
        x_test = setup_x(M, "test")
        np.savetxt("x_test.txt", x_test)
    else:
        x_test = loadtxt("x_test.txt");
    print("x_test is set up")
    
    if not os.path.exists("y_test.txt"):
        y_test = setup_y(M, "test")
        np.savetxt("y_test.txt", y_test)
    else:
        y_test = loadtxt("y_test.txt")
    print("y_test is set up")
    test_size = int(y_test.shape[1]) # there are 10000 sample in test set
    
    w0 = np.random.normal(0.0, 1.0, (x.shape[0]+1, y.shape[0]))/math.sqrt(x.shape[0] * y.shape[0])
    alpha = 0.00001;
    max_iter = 5000;
    # Run gradient descent if haven't done yet
    if not os.path.exists("part4_w.txt"):
        w = grad_descent(NLL, NLL_gradient, x, y, w0, alpha, max_iter)
    w_list = np.loadtxt("part4_w.txt").reshape((11,x.shape[0]+1,y.shape[0]))
    x_axis = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
    train_performance = []
    test_performance = []
    for i in range (0, 11):
        # training set performance
        acc = check_performance(x, y, w_list[i], training_size)
        train_performance.append(acc)
        
        # test set performance
        acc_t = check_performance(x_test, y_test, w_list[i], test_size)
        test_performance.append(acc_t)
    plt.ylim(0,110)
    plt.plot(x_axis, train_performance, label="training")
    plt.plot(x_axis, test_performance, label="validation")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig("part4.png")
    
def check_performance(x, y, w, set_size):
    x_hat = vstack((ones((1, x.shape[1])), x))
    y_hat = dot(w.T, x_hat)
    result = np.argmax(y_hat, axis = 0)
    correct = set_size - np.count_nonzero(np.argmax(y, axis = 0)-result)
    acc = correct * 1.0 / set_size * 100
    return acc

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
def NLL(x, y, w):
    x = vstack( (ones((1, x.shape[1])), x)) # add the "b"
    p = lin_combin(w, x)
    return -sum(y*log(p)) 
    
    
def NLL_gradient(x, y, w):
    x = vstack( (ones((1, x.shape[1])), x)) # add the "b"
    p = lin_combin(w, x)
    return dot(x, (p - y).T)
    
    
def grad_descent(f, df, x, y, init_w, alpha, max_iter):
    EPS = 1e-10   #EPS = 10**(-10)
    prev_w = init_w-10*EPS
    w = init_w.copy()
    iter = 0
    ws = np.array([w])
    while norm(w - prev_w) >  EPS and iter < max_iter:
        prev_w = w.copy()
        w -= alpha*df(x, y, w)
        iter += 1
        if (iter % 100 == 0):
            print("iter", iter, "NLL(x)", f(x, y, w))
        if (iter % 500 == 0):
            ws = vstack((ws, [w]))
    np.savetxt("part4_w.txt", ws.flatten())
    return w


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    
    
# follow the diagram
# for part 2, W should be 784 x 10. b should be 10 x 1
def lin_combin(w, x):
    #o = (dot(w.T, x) + b)
    o = dot(w.T, x)
    return softmax(o)


def finite_diff_gradient(x, y, w, b, i, j, h):
    """
    Use finite difference to calculate the gradient on w_ij with h.
    x: input
    y: expected output
    w: weight
    b: bias
    """
    
    c = NLL(x, y, w, b) 
    new_w = w
    new_w[i, j] += h
    new_c = NLL(x, y, new_w, b) 
    return (new_c - c)/h


def test_gradient(x, y, b, w):
    """
    Compare the result of my gradient function NLL_gradient and the gradient get 
    from finite differences method. Test three times on different w_ij.
    """
    
    j_list = [5, 8, 8]
    i_list = [300, 601, 600]
    
    h = 10e-5
    
    for k in range(0, 3):
        i = i_list[k]
        j = j_list[k]
        g_finite = finite_diff_gradient(x, y, w, b, i, j, h)
        g_mine = NLL_gradient(x, y, w, b)[i, j]
        diff = abs(g_mine - g_finite)
        print("Test "+str(k)+":")
        print("gradient_finite_differences = "+str(g_finite))
        print("gradient_my_function = "+str(g_mine))
        print("difference = "+str(diff))
        print("-------------------------------")     


# x should be 784 x 5000
# y should be 10 x 5000
# b should be same as y
# w should be 784 x 10
def setup_x(M, set):
    x = np.zeros((784,), dtype=float) # dummy row. will delete later
    for digit in range(0,10):
        set_name = set + str(digit)
        for i in range(0, len(M[set_name])):
            x = vstack((x, M[set_name][i]))
    
    x = np.delete(x, (0), axis=0) #delete dummy row
    return x.T


def setup_y(M, set):
    y = np.zeros((10,))
    for digit in range(0,10):
        set_name = set + str(digit)
        z = np.zeros((10,))
        z[digit] = 1
        for i in range(0, len(M[set_name])):
            y = vstack((y, z))
    y = np.delete(y, (0), axis=0)
    return y.T


"""
calling main function.
"""
if __name__ == '__main__':
    main()


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
