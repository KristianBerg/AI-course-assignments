"""Various helper functions and data initializations"""
import aiio
import numpy as np
import random as rand
from matplotlib import pyplot as plt

(labels, datapoints) = aiio.to_batch(aiio.input_dense_LIBVSM('data'))

def normalize(feats):
    matr = feats.copy(order='F');
    matr /= matr.max()
    return matr

feats3 = normalize(datapoints)
feats2 = normalize(np.vstack([feats3[0,:], feats3[2,:] / feats3[1,:]]))

def train_stoch(perceptron,
                labels,
                feats,
                error_max = 0,
                epoch_max = 100,
                learn_rate = 0.1):
    """Do a stochastic training run for a unit"""
    for i in range(epoch_max):
        perm = np.random.permutation(np.vstack([labels,feats]).T)
        for row in perm:
            perceptron.train(row[0], row[1:], learn_rate / len(feats[0]))
        if np.abs(labels - perceptron.batch_classify(feats)).sum() \
           <= error_max:
            return

def train_batch(logit,
                labels,
                feats,
                error_max = 0,
                epoch_max = 100,
                learn_rate = 0.1):
    """Do a batch training run for a unit (batch training only implemented
    for logit units)
    """
    for i in range(epoch_max):
        logit.batch_train(labels, feats, 1)
        if np.abs(labels - logit.batch_classify(feats)).sum() \
           <= error_max:
            return
        
def plot_prediction(labels, feats):
    classfilter = lambda c: \
                  np.transpose([feats[1:,i] for i in range(len(labels)) if labels[i] == c])
    red = classfilter(0)
    blue = classfilter(1)
    plt.plot(red[0], red[1], 'r.')
    plt.plot(blue[0], blue[1], 'b.')    
    
def plot_3wts(weights, bias_unit):
    line = [[0,1], [-weights.dot([bias_unit,0,0])/weights[2],
                    - weights.dot([bias_unit,1,0])/weights[2]]]
    plt.plot(line[0], line[1])

def plot_2wts(weights):
    plot_3wts(np.vstack(weights, ones(1)))

def plot(perceptron):
    """Plot the datapoints and the line discriminated by a 3-weight unit"""
    plot_prediction(labels, feats3)
    plot_3wts(perceptron.weights, feats3[0][0])

def error(perceptron, labels, features):
    return np.abs(labels - perceptron.batch_classify(features)).sum()
    
    
def iter(perceptron, learn_rate):
    """Run 100 training iterations for a unit, or until zero error, plot lines
    throughout
    """
    plt.ion()
    fig = plt.figure()
    for i in range(100):
        train_stoch(perceptron, labels, feats3, epoch_max=1, learn_rate = learn_rate)
        fig.canvas.draw()
        plot(perceptron)
        if error(perceptron, labels, feats3) == 0:
            print(repr(i) + " epochs")
            break
    
        
