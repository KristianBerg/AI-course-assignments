import perceptron
import numpy as np

class Logit(perceptron.Perceptron):
    def __repr__(self):
        return "1-layer logit net, weights: " + repr(self.weights)
    
    def batch_train(self, labels, examples, learn_rate):
        self.weights += (learn_rate / len(labels)) * examples.dot(labels - 1 \
            / (np.ones(len(labels)) + np.exp(-self.weights.dot(examples))))

    def train(self, label, example, learn_rate = 1):
        self.weights += learn_rate * example * (label - 1 \
            /  np.exp(-self.weights.dot(example)))
