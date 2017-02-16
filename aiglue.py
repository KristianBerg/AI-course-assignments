import aiio
import numpy
import random as rand
import matplotlib.pyplot as pyplot

(labels, datapoints) = aiio.to_batch(aiio.input_dense_LIBVSM('data'))

def normalize(feats):
    matr = feats.copy(order='F');
    matr[1:,:] /= matr[1:,:].max()
    return matr

feats3 = normalize(datapoints)
feats2 = normalize(numpy.vstack([feats3[0,:], feats3[2,:] / feats3[1,:]]))

def train_perceptron(perceptron, labels, feats, iter_count):
    for t in range(1,iter_count):
        ex = rand.randint(0, len(labels)-1)
        perceptron.train(labels[ex], feats[:,ex], 1000/(1000 + t))

def plot_prediction(labels, feats):
    classfilter = lambda c: \
                  numpy.transpose([feats[1:,i] for i in range(len(labels)) if labels[i] == c])
    red = classfilter(0)
    blue = classfilter(1)
    pyplot.plot(red[0], red[1], 'r.')
    pyplot.plot(blue[0], blue[1], 'b.')

    
def plot_weights(weights):
    line = numpy.transpose([[0,weights.dot([1,0])], [1, weights.dot([1,1])]])
    pyplot.plot(line[0], line[1])

def plot(perceptron):
    plot_prediction(labels, feats3)
    plot_weights(perceptron.weights)
    pyplot.show()

def iter(perceptron):
    train_perceptron(perceptron, labels, feats2, 1000)
    plot(perceptron)
