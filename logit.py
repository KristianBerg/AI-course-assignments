import perceptron
import numpy as np

class Logit(perceptron.Perceptron):
    """Logit unit, extends Perceptron with different weight update.
    """
    def __repr__(self):
        return "1-layer logit net, weights: " + repr(self.weights)
    
    def batch_train(self, labels, examples, learn_rate):
        """Run weight update on a matrix of examples"""
        self.weights += ((learn_rate / len(labels)) * 
            examples.dot( 
                labels - 1.0 / 
                (np.ones(len(labels)) + np.exp(-self.weights.dot(examples))) 
            ))

    def train(self, label, example, learn_rate = 1):
        """Run weight update on a single example"""
        self.weights += (learn_rate * example * (
            label - 1.0 / (1 + np.exp(-self.weights.dot(example)))))
