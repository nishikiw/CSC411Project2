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
import time
import random as rn

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
-distribution of number of data in training/test/val
-number of hidden units
-lamda. Used for regularization (in part 8)
-number of iterations
-number of data in the mini-batch
"""

np.random.seed(0)
rn.seed(0)

act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

# helper function to read the images
def getArray (str):
    im = imread(str)
    return(np.array([im.flatten()]))

# this is used to get a mini batch from the training dataset
def get_train_batch(n, x_train, y_train): # n = image per actor in batch
    x = zeros((0,1024))
    y = zeros((0,6))
    
    idx = rn.sample(range(x_train.shape[0]), n)
    
    for k in range(n):
        x = vstack((x, x_train[idx[k]]))
        y = vstack((y, y_train[idx[k]]))
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

if not os.path.exists("cropped"):
    print("Downloading images")
    download_image(act)

if not os.path.exists("part7_test"):
    print("Set up training, validation and test set")
    split_set(act, 75, 15, 30) # training, val, test

"""
Set up architecture of the network
"""
# placeholder means the var won't change during runtime
x = tf.placeholder(tf.float32, [None, 1024])

# variable means the var will be changed during runtime
nhid = 700 # number of hidden units
alpha = 0.00001
max_iter = 30000
mini_batch_size = 20
lam = 0.005 # right now lamda is set to 0. Need to change it for part 8?

"""
# At iteration 9500, Results of changing: 
nhid: 
(100, 0.877, 37), (200, 0.888, 6.47), (300, 0.9, 2.29), (400, 0.9, 1.12), (500, 0.9, 0.6)
-> pick 300 hidden units b/c one of the best results and less unit = less chance of overfitting

mini_batch:
(20, 0.9, 12.7), (30, 0.88, 3.37) (40, 0.88, 2.22), (50, 0.9, 2.29), (60, 0.9, 1.54), (70, 0.9, 1.34)
-> pick 50 b/c it has 0.9 performance and not too big/small size

parameters used for part 7:
nhid = 300 # number of hidden units
alpha = 0.00001
max_iter = 2000
mini_batch_size = 50
lam = 0.0000 
plot from 0 to 2000, every 200

"""

"""
W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
"""

W0 = tf.Variable(np.random.normal(0.0, 0.1, (1024, nhid)).astype(float32)/math.sqrt(1024 * nhid))
b0 = tf.Variable(np.random.normal(0.0, 0.1, (nhid)).astype(float32)/math.sqrt(nhid))

W1 = tf.Variable(np.random.normal(0.0, 0.1, (nhid, 6)).astype(float32)/math.sqrt(6 * nhid))
b1 = tf.Variable(np.random.normal(0.0, 0.1, (6)).astype(float32)/math.sqrt(6))

layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])


# regularization/penalty
# according to class, weight penalty is used when the network is overfitting
# to create overfitting in this current architecture, can increase number of neurons (nhid)?

decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(alpha).minimize(reg_NLL)

"""
Done setting up architecture. Actually run the network now
"""
# init will init W0, B0, W1, B1 to random value
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_test, y_test = get_whole_set(act, "part7_test")
x_val, y_val = get_whole_set(act, "part7_validation")
x_train, y_train = get_whole_set(act, "part7_training")

x_axis = np.arange(11) * 200
train_performance = []
test_performance = []
val_performance = []

for i in range(max_iter+1):
  batch_xs, batch_ys = get_train_batch(mini_batch_size, x_train, y_train) # <-change size of mini batch here. Max is 75 for now
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  
  if i % 1000 == 0:
    print ("i=",i)
    print ("Cost:", sess.run(reg_NLL, feed_dict={x: x_train, y_:y_train}))
    acc_tr = sess.run(accuracy,feed_dict={x: x_train, y_: y_train})
    train_performance.append(acc_tr * 100)
    print ("Train:", acc_tr)
    
    acc_t = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    test_performance.append(acc_t * 100)
    print ("Test:", acc_t)

    acc_v = sess.run(accuracy,feed_dict={x: x_val, y_: y_val})
    val_performance.append(acc_v * 100)
    print ("Validation:", acc_v)

    print ("Penalty:", sess.run(decay_penalty))

"""
plt.ylim(0,110)
plt.plot(x_axis, test_performance, label="test")
plt.plot(x_axis, train_performance, label="training")
plt.plot(x_axis, val_performance, label="validation")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Iteration')
plt.ylabel('Correctness(%)')
plt.savefig("part7.png")
"""
