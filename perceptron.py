import aiio
import numpy as num

class Perceptron:
    def __init__(self, init_wts):
        self.weights = num.array(init_wts).reshape(1,-1)
        
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Perceptron, weights: " + repr(self.weights)

    def train(self, label, example):
        prediction = int(num.dot(self.weights, example) > 0)
        if prediction == label:
            return
        if prediction != label:
            self.weights += ((label - prediction) * example)
            return
        

    def classify(self, example):
        return int(num.dot(self.weights, example) > 0)


    def batch_train(self, labels, examples):        
        result = num.rint(num.greater(self.weights.dot(examples), 0))
        self.weights += num.sum(examples.dot(labels.T - result.T), axis=1)
