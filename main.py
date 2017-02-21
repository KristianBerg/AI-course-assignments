import aiglue
import perceptron
import logit
import numpy as np
import matplotlib.pyplot as plt

tron1 = perceptron.Perceptron(3)
log1 = logit.Logit(3)

n_epoch_tron = 0
while True:
    aiglue.train_stoch(tron1, aiglue.labels, aiglue.feats3, learn_rate = 0.1)
    n_epoch_tron += 1
    if np.abs(aiglue.labels - tron1.batch_classify(aiglue.feats3)).sum() == 0:
        break

n_epoch_logit = 0
while True:
    old_wts = log1.weights.copy()
    aiglue.train_stoch(log1, aiglue.labels, aiglue.feats3, learn_rate = 8)
    n_epoch_logit += 1
    if np.linalg.norm(log1.weights - old_wts) < 0.001:
        break

plt.subplot(211)
aiglue.plot(tron1)
plt.title('Perceptron (' + repr(n_epoch_tron) + ' epochs)')
plt.subplot(212)
aiglue.plot(log1)
plt.title('Logistic Regression (' + repr(n_epoch_logit) + ' epochs)')
plt.show()
