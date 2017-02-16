import aiio
import numpy as num
import random as rand

class Perceptron:
    def __init__(self, init_wts):
        if isinstance(init_wts, int):
            self.weights = num.array([rand.random() for _ in range(init_wts)])
        else:
            self.weights = num.array(init_wts).reshape(1,-1).astype(float)
        
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Perceptron, weights: " + repr(self.weights)

    def classify(self, example):
        return int(self.weights.dot(example) > 0)
    
    def train(self, label, example, learn_rate = 1):
        example = num.array(example)
        self.weights += (learn_rate * (label - self.classify(example)) * example)

    def batch_classify(self, examples):
        return num.rint(num.greater(self.weights.dot(examples), 0))
