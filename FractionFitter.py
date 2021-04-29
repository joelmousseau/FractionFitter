import numpy as np
import scipy as sp
from scipy import optimize
import math

def findPrediction(x, factors, simulation, data):
	sum = 0.0
	#loop over templates
	print x
	it = np.nditer([factors, simulation], flags=['common_dtype'])
	for factor, sim in it:
		sum += factor*sim / (1 + x*factor)
	if(x == 1.0):
	 	x -= 0.1
	const = data /(1 - x)
	return const - sum	

def predictionDerivative(x, factors, simulation, data):
	sum = 0.0
	it = np.nditer([factors, simulation], flags=['common_dtype'])	
	for factor, sim in it:
		sum -= math.pow(factor,2)*sim / math.pow((1 + x*factor),2)
	if(x == 1.0):
	 	x -= 0.1
	const = -1.0*data / math.pow((1 - x),2)
	return const - sum

#Fitter expects lists in the form of: [ bin1: [template1, template2, template3], bin2: [template1, template2, template3], ... binN: [template1, template2, tempalte3]]
#However, a more natural structure one would use to make a plot of is: [ template1: [bin1, bin2, ... binN], template2: [bin1, bin2, ... binN], template3: [bin1, bin2, ... binN]]
#This function transforms the former into the later. Clearly not needed if the prediction is in the second form to begin with
def shuffleBins(simulation):
	return list(np.stack(simulation, axis=1))

def getLL(factors, simulation, data):
	bigLL = 0.0
	#Loop over bins
	it = np.nditer(data, flags=['c_index'])
	while not it.finished:
		noData = it[0] 
		#print("%d <%d>" % (it[0], it.index) )
		ti = optimize.newton(findPrediction, 0.0, predictionDerivative, args=(factors, simulation[it.index], noData) )
		fi = noData / (1 - ti)
		bigLL += noData*math.log(fi)
		bigLL -= fi
		itT = np.nditer([factors[it.index], simulation[it.index]])
		for factor, sim in itT:
			binPrediction = sim /(1+ factor*ti)
			bigLL += sim*math.log(binPrediction)
			bigLL -= binPrediction

		it.iternext()

	return -1.0*bigLL	

'''
data        = np.array([10.0, 20.0, 30.0, 14.0, 2.0])
factors     = np.array([1.0, 1.0, 1.0, 1.0]) 
simulation  = (np.array([3.0, 5.0, 2.0, 3.0]), np.array([3.0, 12.0, 5.0, 6.0]), np.array([5.0, 12.0, 6.0, 5.0]), np.array([1.0, 2.0, 4.0, 3.0]))
shuffledSim = shuffleBins(simulation)
getLL(factors, shuffledSim, data)
'''
