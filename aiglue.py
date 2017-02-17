import aiio
import numpy as np
import random as rand
from matplotlib import pyplot as plt
import time

(labels, datapoints) = aiio.to_batch(aiio.input_dense_LIBVSM('data'))

def normalize(feats):
    matr = feats.copy(order='F');
    matr[1:,:] /= matr[1:,:].max()
    return matr

feats3 = normalize(datapoints)
feats2 = normalize(np.vstack([feats3[0,:], feats3[2,:] / feats3[1,:]]))

learn_rate = 0.001

def train_stoch(perceptron, labels, feats, error_max = 0.1, epoch_max = 100):
    for _ in range(epoch_max):
        perm = np.random.permutation(np.vstack([labels,feats]).T)
        for row in perm:
            perceptron.train(row[0], row[1:], learn_rate / len(feats))
        """if np.abs(labels - perceptron.batch_classify(feats)).sum() \
           <= error_max * len(labels):
            return"""

def train_batch(logit, labels, feats, error_max):
    while True:
        logit.batch_train(labels, feats, 1)
        if np.abs(labels - logit.batch_classify(feats)).sum() \
            <= error_max * len(labels):
            return
        
def plot_prediction(labels, feats):
    classfilter = lambda c: \
                  np.transpose([feats[1:,i] for i in range(len(labels)) if labels[i] == c])
    red = classfilter(0)
    blue = classfilter(1)
    plt.plot(red[0], red[1], 'r.')
    plt.plot(blue[0], blue[1], 'b.')    
    
def plot_3wts(weights):
    line = [[0,1], [-weights.dot([1,0,0])/weights[2], -weights.dot([1,1,0])/weights[2]]]
    plt.plot(line[0], line[1])

def plot_2wts(weights):
    plot_3wts(np.vstack(weights, ones(1))

def plot(perceptron):
    plot_prediction(labels, feats3)
    plot_3wts(perceptron.weights)

def error(perceptron, labels, features):
    return np.abs(labels - perceptron.batch_classify(features)).sum()
    
    
def iter(perceptron):
    plt.ion()
    fig = plt.figure()
    for i in range(100):
        train_stoch(perceptron, labels, feats3, epoch_max=1)
        fig.canvas.draw()
        plot(perceptron)
        if error(perceptron, labels, feats3) == 0:
            print(repr(i) + " epochs")
            break
    
        
