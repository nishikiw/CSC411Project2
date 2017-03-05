from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

from get_data import download_image, split_set

#import cPickle

import os
from scipy.io import loadmat
import tensorflow as tf

"""
Hi Catherine!! PLEASE READ THIS FIRST!
The majority of the code for part 7 should be completed. I have left comments
all over the code, so hopefully that can help your understanding of what I am
doing :)

However, I didn't get to test the find the best parameters to the network/play 
around with the network setting. If you have time, maybe you can try running
the network with different parameters?
Here is a list of all the parameters that you can change:
line 94: distribution of number of data in training/test/val
line 106: number of hidden units
line 133: lamda. Used for regularization (in part 8)
line 152: number of iterations
line 153: number of data in the mini-batch

Remember to uncomment line 94 when you are running the first time so the dataset
is splitted properply
"""

np.random.seed(0)

act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

# helper function to read the images
def getArray (str):
    im = imread(str)
    return(np.array([im.flatten()]))

# this is used to get a mini batch from the training dataset
def get_train_batch(act, n): # n = image per actor in batch
    
    x = zeros((0,1024))
    y = zeros((0,6))
        
    for k in range(6):
        name = act[k].split()[1].lower()
        used_image_no = []
        while (len(used_image_no) < n):
            r = random.randint(1, 200)
            if r in used_image_no:
                continue
            filename = name+str(r)+'.'+'jpg'
            try:
                x = vstack((x, getArray("part7_training/" + filename)))
                used_image_no.append(r)
            except:
                pass
        one_hot = zeros(6)
        one_hot[k] = 1
        y = vstack((y, tile(one_hot, (n,1))))        
    return x, y
    

# get whole training set
def get_whole_set(act, file):
    x = zeros((0,1024))
    y = zeros((0,6))
    for k in range(6):
        counter = 0
        name = act[k].split()[1].lower()
        for fn in os.listdir('./' + file):
            if (name in fn):
                x = vstack((x, getArray(file + "/" + fn)))
                counter += 1
        one_hot = zeros(6)
        one_hot[k] = 1
        y = vstack((y, tile(one_hot, (counter,1))))
    return x, y


# download, and split dataset
"""
download_image(act)
split_set(act, 75, 15, 30) # training, val, test
"""


"""
Set up architecture of the network
"""
# placeholder means the var won't change during runtime
x = tf.placeholder(tf.float32, [None, 1024])


# variable means the var will be changed during runtime
nhid = 300 # number of hidden units
W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

"""
TODO delete
cPickle is just used to load a saved data, just like np.savetxt/loadtxt
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = tf.Variable(snapshot["W0"])
b0 = tf.Variable(snapshot["b0"])
W1 = tf.Variable(snapshot["W1"])
b1 = tf.Variable(snapshot["b1"])
"""

layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])


# regularization/penalty
# according to class, weight penalty is used when the network is overfitting
# to create overfitting in this current architecture, can increase number of neurons (nhid)?
lam = 0.00000 # right now lamda is set to 0. Need to change it for part 8?
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

"""
Done setting up architecture. Actually run the network now
"""
# init will init W0, B0, W1, B1 to random value
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
x_test, y_test = get_whole_set(act, "part7_test")


for i in range(5000):
  batch_xs, batch_ys = get_train_batch(act, 60) # <-change size of mini batch here. Max is 75 for now
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  
  if i % 100 == 0:
    print ("i=",i)
    print ("Test:", sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
    
    x_val, y_val = get_whole_set(act, "part7_validation")
    print ("Validation:", sess.run(accuracy,feed_dict={x: x_val, y_: y_val}))

    
    batch_xs, batch_ys = get_whole_set(act, "part7_training")
    print ("Train:", sess.run(accuracy,feed_dict={x: batch_xs, y_: batch_ys}))
    print ("Penalty:", sess.run(decay_penalty))

    """
    TODO delete
    This is used to save the W's and b's just like np.savetxt
    snapshot = {}
    snapshot["W0"] = sess.run(W0)
    snapshot["W1"] = sess.run(W1)
    snapshot["b0"] = sess.run(b0)
    snapshot["b1"] = sess.run(b1)
    cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))
    """
    