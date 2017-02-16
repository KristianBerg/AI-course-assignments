import aiio
import numpy

(labels, datapoints) = aiio.to_batch(aiio.input_dense_LIBVSM('data'))
feats = normalize(datapoints[0:3:2])

def normalize(feats):
    matr = feats.copy(order='F');
    matr[1:,:] /= matr[1:,:].max()
    return matr
