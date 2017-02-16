import numpy as np;
import random as rand;
import matplotlib.pyplot as plt

def readData(filename):
	dataFile = open(filename)
	dataStr = dataFile.read()
	dataArray = dataStr.split();

	Total = []
	A = []

	for i in range(0, len(dataArray), 2):
		Total.append(float(dataArray[i]))
		A.append(float(dataArray[i+1]))

	return [Total, A]

def scale(data, scalingConst):
	for i in data:
		for j in range(0, len(i)):
			i[j] = i[j] / scalingConst

def batchGradientDescent(y, x):
	w0 = 0
	w1 = 0
	alpha = 0.4
	iterations = 1000
	q = len(y)
	for i in range(1, iterations):
		w0 = w0 + alpha/q * sum(y - x*w1 - w0*np.ones(q))
		w1 = w1 + alpha/q * np.dot(x, y - x*w1 - w0*np.ones(q))
	return [w0, w1]

def stochasticGradientDescent(y, x):
	w0 = 0
	w1 = 0
	alpha = 0.05
	iterations = 1000
	q = len(y)
	order = range(0,q)
	for i in range(0, iterations):
		rand.shuffle(order)
		for j in order:
			w0 = w0 + alpha * (y[j] - x[j]*w1 - w0)
			w1 = w1 + alpha * x[j] * (y[j] - x[j]*w1 - w0)
	return [w0, w1]

def classification(w, c0a, c0total, c1a, c1total):
	q = len(c0a)
	res_c0 = 0
	res_c1 = 0
	for i in range(0, q):
		if(w[0] + w[1]*c0total[i] < c0a[i]):
			res_c0 += 1
		if(w[0] + w[1]*c1total[i] < c1a[i]):
			res_c1 += 1
	return 100 * max((res_c0 + (q - res_c1))/(2.0*q), (res_c1 + (q - res_c0))/(2.0*q))
	
def plotResults(batchEn, batchFr, stochEn, stochFr, enA, enTotal, frA, frTotal):
	batchEnLine = [batchEn[0], sum(batchEn)]
	batchFrLine = [batchFr[0], sum(batchFr)]
	stochEnLine = [stochEn[0], sum(stochEn)] 
	stochFrLine = [stochFr[0], sum(stochFr)]
	xVals = [0, 1]
	
	plt.subplot(211)
	plt.title('Batch method')
	plt.plot(enTotal, enA, 'bs', frTotal, frA, 'r^')
	plt.plot(xVals, batchEnLine, 'g')
	plt.plot(xVals, batchFrLine, 'y')
	plt.subplot(212)
	plt.title('Stochastic method')
	plt.plot(enTotal, enA, 'bs', frTotal, frA, 'r^')
	plt.plot(xVals, stochEnLine, 'g')
	plt.plot(xVals, stochFrLine, 'y')
	plt.show()

[enTotal, enA] = readData("en_a")
[frTotal, frA] = readData("fr_a")

allData = [enTotal, enA, frTotal, frA]
scale(allData, max(max(frTotal), max(enTotal)))
"""
res0 = batchGradientDescent(np.append(enA, frA), np.append(enTotal, frTotal))
print("Batch method weights: " + str(res0))
classificationRes0 = classification(res0, enA, enTotal, frA, frTotal)
print("Percent correctly classified: " + str(classificationRes0) + "%")

res1 = stochasticGradientDescent(np.append(enA, frA), np.append(enTotal, frTotal))
print("Stochastic method weights: " + str(res1))
classificationRes1 = classification(res1, enA, enTotal, frA, frTotal)
print("Percent correctly classified: " + str(classificationRes1) + "%")
"""
batchEn = batchGradientDescent(np.array(enA), np.array(enTotal))
batchFr = batchGradientDescent(np.array(frA), np.array(frTotal))
stochEn = stochasticGradientDescent(np.array(enA), np.array(enTotal))
stochFr = stochasticGradientDescent(np.array(frA), np.array(frTotal))

print("batch english weights: " + str(batchEn))
print("batch french weights: " + str(batchFr))
print("stochastic english weights: " + str(stochEn))
print("stochastic french weights: " + str(stochFr))

for i in range(0, len(allData)):
	allData[i].sort()
plotResults(batchEn, batchFr, stochEn, stochFr, enA, enTotal, frA, frTotal)

