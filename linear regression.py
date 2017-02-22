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
	oldW = [w0, w1]
	alpha = 0.4
	maxEpochs = 10000
	q = len(y)
	for i in range(1, maxEpochs):
		w0 = w0 + alpha/q * sum(y - x*w1 - w0*np.ones(q))
		w1 = w1 + alpha/q * np.dot(x, y - x*w1 - w0*np.ones(q))
		"""
		diffLen = np.sqrt(np.power(w0-oldW[0], 2) + np.power(w1-oldW[1], 2))
		currentLen = np.sqrt(np.power(w0, 2) + np.power(w1, 2))
		if(diffLen < 1e-4):
			print("number of batch epochs: " + str(i))
			return [w0, w1]
		oldW = [w0, w1]	 
		"""
	return [w0, w1]

def stochasticGradientDescent(y, x):
	w0 = 0
	w1 = 0
	oldW = [w0, w1]
	alpha = 0.05
	q = len(y)
	maxEpochs = 10000
	order = range(0,q)
	for i in range(0, maxEpochs):
		rand.shuffle(order)
		for j in order:
			w0 = w0 + alpha * (y[j] - x[j]*w1 - w0)
			w1 = w1 + alpha * x[j] * (y[j] - x[j]*w1 - w0)
		"""
		diffLen = np.sqrt(np.power(w0-oldW[0], 2) + np.power(w1-oldW[1], 2))
		currentLen = np.sqrt(np.power(w0, 2) + np.power(w1, 2))
		if(diffLen < 1e-4):
			print("number of stochastic epochs: " + str(i))
			return [w0, w1]
		oldW = [w0, w1]
		"""
	return [w0, w1]


	
def plotResults(batchEn, batchFr, stochEn, stochFr, enA, enTotal, frA, frTotal):
	endPoint = 80000
	batchEnLine = [batchEn[0], batchEn[0] + endPoint*batchEn[1]]
	batchFrLine = [batchFr[0], batchFr[0] + endPoint*batchFr[1]]
	stochEnLine = [stochEn[0], stochEn[0] + endPoint*stochEn[1]] 
	stochFrLine = [stochFr[0], stochFr[0] + endPoint*stochFr[1]]
	xVals = [0, endPoint]
	
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

scaleConst = max(max(enTotal), max(frTotal))
scale(allData, scaleConst)

batchEn = batchGradientDescent(np.array(enA), np.array(enTotal))
batchFr = batchGradientDescent(np.array(frA), np.array(frTotal))
stochEn = stochasticGradientDescent(np.array(enA), np.array(enTotal))
stochFr = stochasticGradientDescent(np.array(frA), np.array(frTotal))
allResults = [batchEn, batchFr, stochEn, stochFr]

[enTotal, enA] = readData("en_a")
[frTotal, frA] = readData("fr_a")
for i in range(0, len(allResults)):
	j = 0
	allResults[i][0] *= scaleConst

print("batch english weights: " + str(batchEn))
print("batch french weights: " + str(batchFr))
print("stochastic english weights: " + str(stochEn))
print("stochastic french weights: " + str(stochFr))

plotResults(batchEn, batchFr, stochEn, stochFr, enA, enTotal, frA, frTotal)

