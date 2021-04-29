import numpy as np
from bmlfit import BMLFit

data        = np.array([10.0, 20.0, 30.0, 14.0]) #[bin 0, bin1, bin2, ... binN]
initialFactors     = np.array([1.0, 1.0, 1.0, 1.0]) #[template 0, template 1, tempalte 2, ... templateN]
simulation  = np.array([np.array([3.0, 5.0, 2.0, 3.0]), np.array([3.0, 12.0, 5.0, 6.0]), np.array([5.0, 12.0, 6.0, 5.0]), 
	np.array([1.0, 2.0, 4.0, 3.0]) ] ) #[Bin 0 [template 0, template 1, template 2, .. templateN], Bin 1 [template 0, template 1, template 2, .. templateN], ... Bin N [template 0, template 1, template 2, .. templateN],]
weights     = np.array([np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0]),  np.array([1.0, 1.0, 1.0, 1.0]),  
	np.array([1.0, 1.0, 1.0, 1.0]) ] ) #[Bin 0 [template 0, template 1, template 2, .. templateN], Bin 1 [template 0, template 1, template 2, .. templateN], ... Bin N [template 0, template 1, template 2, .. templateN],]

fitter = BMLFit(simulation, data, weights)
fitter.do_fit()
print (fitter.get_fit_result())
#print (fitter.get_true_templates())
print(fitter.get_chi2(True))


