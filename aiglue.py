import aiio
import numpy
import random as rand

(labels, datapoints) = aiio.to_batch(aiio.input_dense_LIBVSM('data'))

def normalize(feats):
    matr = feats.copy(order='F');
    matr[1:,:] /= matr[1:,:].max()
    return matr

feats2 = normalize(datapoints[0:3:2])
feats3 = normalize(datapoints)

def trainperceptron(perceptron, labels, feats, iter_count):
    for _ in range(1,iter_count):
        ex = rand.randint(0, len(labels)-1)
        perceptron.train(labels[ex], feats[:,ex])
