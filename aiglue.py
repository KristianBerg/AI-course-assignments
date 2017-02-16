import aiio
import numpy
import random as rand
import matplotlib.pyplot as pyplot

(labels, datapoints) = aiio.to_batch(aiio.input_dense_LIBVSM('data'))

def normalize(feats):
    matr = feats.copy(order='F');
    matr[1:,:] /= matr[1:,:].max()
    return matr

feats2 = normalize(datapoints[0:3:2])
feats3 = normalize(datapoints)

def train_perceptron(perceptron, labels, feats, iter_count):
    for _ in range(1,iter_count):
        ex = rand.randint(0, len(labels)-1)
        perceptron.train(labels[ex], feats[:,ex])

def plot_prediction(labels, feats):
    classfilter = lambda c: \
                  [feats[1:,i] for i in range(len(labels)) if labels[i] == c]
    red = classfilter(0)
    blue = classfilter(1)
    pyplot.plot(red[0], red[1], 'r.')
    pyplot.plot(blue[0], blue[1], 'b.')
    pyplot.show()
