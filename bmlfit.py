import numpy as np
import scipy as sp
from scipy import optimize
import math

#Define the Fitting Exception Class
class FittingError(Exception):
    pass

class BMLFit:
   
  

  def __init__(self, sim, data, wgts=np.array([]), drop=np.array([]) ):
    #The weights array (if provided) must have the same shape as the simulation
    #If not, raise an exception
    if(wgts.size > 0 and wgts.shape != sim.shape):
       raise ValueError("Weight array must have same shape as MC sources!")

    #If weights are not provided, set them all to 1.0
    if wgts.size == 0:
        self.weights = np.full(sim.shape, 1.0)
    else:
       self.weights = wgts

    #Create an array of the "true" simulation, same shape as the simulation itself
    self._Aij = np.empty(sim.shape) #"A" ij is the name of the variable from the original paper
    self.simulation = sim
    self.data = data
    #Create an intial array of factors the fit is trying to measure
    #Nominally these should be one, but the user can override it later
    self.factors = np.full(sim[0].size, 1.0)
    self.result = None
    self.constraints = [(0,1.0)] * sim.shape[0]
    print (self.constraints)
    if(drop.size > 0 and drop.shape != data.shape):
       raise ValueError("Array of omitted bins must have same shape as data bins!")
    if drop.size == 0:
       self.dropBins = np.full(data.shape, False)
    else:
       self.dropBins = drop      
  
 
  def _find_prediction(self, x, data, bin, wgtFactors):
    sum = 0.0
	  #loop over templates
    it = np.nditer([wgtFactors, self.simulation[bin]], flags=['common_dtype'])
    #print ("x is: %.2f" % x)
    #print ("data is: %.2f" % data)
    for factor, sim in it:
       #print ("Factor is: %.2f" % factor)
       #print ("Sim is: %.2f" % sim)
       sum += factor*sim / (1 + x*factor)
    const = data /(1 - x)
    return sum - const	

  def _prediction_derivative(self, x, data, bin, wgtFactors):
    sum = 0.0

    it = np.nditer([wgtFactors, self.simulation[bin]], flags=['common_dtype'])
    '''
    if(x == 1.0):
       return -50.0
    '''
    for factor, sim in it:
       sum -= math.pow(factor,2)*sim / math.pow((1 + x*factor),2)  
    const = -1.0*data / math.pow((1 - x),2)
    return sum - const

  def _prediction_second_derivative(self, x, data, bin, wgtFactors):
    sum = 0.0

    it = np.nditer([wgtFactors, self.simulation[bin]], flags=['common_dtype'])
    '''
    if(x == 1.0):
       return -50.0
    '''
    for factor, sim in it:
       sum += 2.0*math.pow(factor,3)*sim / math.pow((1 + x*factor),3)  
    const = 2.0*data / math.pow((1 - x),2)
    #return const - sum
    return sum - const
  
  def _get_log_L(self, x):
    bigLL = 0.0   
    
    #SciPy wants the objective functions optimization parameter to be 'x'
    #For the sake of readabiliyt, I re-define this paramter is 'objective' to make the code more transparent
    objective = x 
	  #Loop over bins
    it = np.nditer(self.data, flags=['c_index'])
    while not it.finished:
        noData = it[0]
        i = it.index #This indexes the current bin
        ti = 0.0
        fi = 0
        binPredictionIfZero = 0.0
        if(self.dropBins[i]):
           continue

        wgtFactors = np.multiply(self.weights[i], objective)
        maxFrac = np.amax(wgtFactors)
        maxSource = np.argmax(wgtFactors) #index of the largest source
        #Determine if thee are multiple templates with the "largest fractions"
        isMax = np.equal(wgtFactors, np.full(self.weights[i].size, maxFrac))
        isNotMax = np.invert(isMax)
        nMax = np.sum(isMax)
        #Current version of numpy doesn't support masking elements in np.sum. *sad trombone* 
        largetstTemplateIntegral = np.sum(self.simulation[i][isMax] ) # only if isMax entry is true 
		    #Special case if there are no data events in the bin see eqn 22 in Barlow and Beeston
        if( noData == 0):
            ti = 1.0
            binPredictionIfZero = 0.0
            maxSource = -1
        
        elif (largetstTemplateIntegral == 0.0):
            binPredictionIfZero = noData / (1 + maxFrac)
            # A_ki -= a_ji[par] * wgtFrac[par] / (maxWgtFrac - wgtFrac[par]); loop over par
            binPredictionIfZero = binPredictionIfZero - np.sum(np.divide(np.multiply(self.simulation[i], wgtFactors), np.add(-1.0*wgtFactors, maxFrac) )[isNotMax] )
            if(binPrediction > 0):
               #divide by the number of sources with zero events
               #This is because later in the code we will add all these together (I think) and we need to re-normalize that sum
               binPredictionIfZero /= nMax 
               ti =  -1.0 / (maxFrac)  
        else:
            #ti = optimize.newton(self._find_prediction, ti, fprime=self._prediction_derivative, args=(noData,i , wgtFactors, ), fprime2=self._prediction_second_derivative )
            ti = optimize.newton(self._find_prediction, ti, args=(noData, i, wgtFactors, ) )
            maxSource = -1 #not needed in this case

        #should be this? fFractions[mc]*weight*binPrediction
        #Loop over templates
        itT = np.nditer([wgtFactors, self.simulation[i]], flags=['c_index'])
        for factor, sim in itT:
            j = itT.index #This indexes the current template
            #This triggers if at least one mc template has 0 events in a bin AND
            #That template has the largest weight (or is 'tied' for largest weight)
            #In the event this is true for multiple templates, the LL is penalized by -ln(nMax)  
            if(maxSource > 0 and factor == wgtFactors[maxSource]):
               binPrediction = binPredictionIfZero
            else:
               binPrediction = sim /(1+ factor*ti)
            
            self._Aij[i][j] = binPrediction 
            fi += binPrediction*factor
            if(binPrediction > 0 and sim > 0):
               bigLL += sim*math.log(binPrediction)
            bigLL -= binPrediction #Divided by nMax
        if(noData > 0 and fi > 0):
           bigLL += noData*math.log(fi)
        bigLL -= fi 
        it.iternext()
    return -1.0*bigLL	
  
  #Class expects lists in the form of: [ bin1: [template1, template2, template3], bin2: [template1, template2, template3], ... binN: [template1, template2, tempalte3]]
  #However, a more natural structure one would use to make a plot of is: [ template1: [bin1, bin2, ... binN], template2: [bin1, bin2, ... binN], template3: [bin1, bin2, ... binN]]
  #These functions transform the former into the later. Clearly not needed if the prediction is in the second form to begin with
  def shuffle_bins(self):
    self.simulation = self.shuffle(self.simulation)
    self.weights    = self.shuffle(self.weights)
    self._Aij = np.empty(self.simulation.shape)

  def shuffle(self,bins):
    return np.array(np.stack(bins, axis=1))
  
  def set_intial_factors(self, init):
    if(init.size != self.simulation[0].size):
       raise ValueError("Array of intial factors must have same shape as MC sources!")
    self.factors = init
  
  def set_constraints(self, init):
    self.constraints = init

  def get_fit_result(self):
    if(self.result is None):
        raise FittingError("Please run the fit by executing 'do_fit' before checking for the result")   
    elif(self.result.success):
        return self.result.x
    else:
        raise FittingError("Fit did not termiante successfully, no result available!")    

  def get_chi2(self, divideByDof=True):
    nDof = 1.0
    if(self.result is None):
        raise FittingError("Please run the fit by executing 'do_fit' before checking for the goodness of fit") 
    includedBins = np.logical_not(self.dropBins) #Only Include bins in the chi2 we did not drop
    simResult = np.sum(np.multiply(self.weights, np.multiply(self.simulation, self.factors) ), axis=1)
    logAij    = np.log(self._Aij)
    logSim    = np.log(self.simulation)
    logLyn = np.sum(np.multiply(self.data, np.log(simResult)) - simResult )
    logLmn = np.sum(np.multiply(self.data, np.log(self.data)) - self.data )
    logLyn += np.sum(np.subtract(np.multiply(self.simulation, logAij), self._Aij) )
    logLmn += np.sum(np.subtract(np.multiply(self.simulation, logSim), self.simulation) )
    if(divideByDof):
        nDof = self.factors.size 

    return (2*logLmn - 2*logLyn) / nDof
  
  #Gettor function for private Aij
  #Normally not needed by the user, but provided if need be
  def get_true_templates(self):
    return self._Aij


  def do_fit(self):
    limits = ((0, 1.0), (0, 1.0), (0, 1.0))
    #solver = 'Nelder-Mead'
    #solver = 'SLSQP'
    solver = 'BFGS'
    self.result = optimize.minimize(self._get_log_L, self.factors, method=solver
      #, bounds=limits
      )
    #Change the factors to the result of the fit
    self.factors = self.result.x
    
 



