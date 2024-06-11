import numpy as np
from scipy.linalg import sqrtm

def wmean(weight, x):
	""" Perform Step 4 in PR book Table 3.4

	Args:
		weight (array): vector that stores weights for each unscented point to calculate mean estimate
		x (array)): matrix for all sample unscented points (could be contact position or sensor)
	
	Return:
		Weighted sum of some variable considering all unscented points
	"""
	return x.dot(weight)

def wcov(weight, x, y):
	""" Perform Step 5 in PR book Table 3.4

	Args:
		weight (array): vector that stores weights for each unscented point to calculate covariance matrix
		x (array): matrix for all sample unscented points (could be contact position or sensor)
		y (array): matrix for all sample unscented points (could be contact position or sensor)
  
	Return:
		Updated Covariance matrix of current step
	"""
    # x(: , i) y(: , i) represent samples
	n = x.shape[1] 			# number of samples 
	mux = wmean(weight, x)	# weighted sum of one variable 
	muy = wmean(weight, y)	# weigthed sum of the other variable
	# Eqn. in Step 5
	return (x - np.expand_dims(mux, 1).dot(np.ones([1, n]))).dot(np.diag(weight).dot((y - np.expand_dims(muy, 1).dot(np.ones([1, n]))).T))
    
def unscentedTrans(lam, mu, sigma):
	""" Perform the sampling step (2 and 6) in PR book Table 3.4

	Args:
		lam (_type_): the gamma parameter in step 2 and 4
		mu (_type_): mean estimate
		sigma (_type_): covariance matrix

	Returns:
		_type_: sampled unscented points
	"""
	n = len(mu)						# dimension of mean estimate -> n
	mu = np.expand_dims(mu, 1)		# mean estimate vector of shape (n, 1)
	m = sqrtm((n+lam) * sigma)		# correponding covariance matrix of (n, n)
	X = mu * np.ones([1, n])		# obtain a matrix where columns are mean estimate
	# stack to get a matrix of unscented points sampled around the mean estimate
	return np.hstack([mu, X + m, X - m])

def error_ellipsoid(mu, sig_sqrtm, P):
	""" Generate an ellipsoid shape of the error (variance) area around mean estiamte

	Args:
		mu (array): mean estimate
		sig_sqrtm (array): covariance matrix
		P (float): parameter control ellipsoid shape
	"""
	theta = np.arange(-np.pi, np.pi, 0.01)
	r = (-2*np.log(1-P))**0.5
	w = np.vstack([r*np.cos(theta), r*np.sin(theta)])
	x = sig_sqrtm.dot(w) + np.repeat(mu.T, len(theta), axis=1)
	return x.T