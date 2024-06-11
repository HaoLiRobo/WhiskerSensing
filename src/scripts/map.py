import numpy as np
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import check_pairwise_arrays, check_array, row_norms, _euclidean_distances_upcast, safe_sparse_dot
from sklearn.datasets import make_classification
import mcubes
import pandas as pd


class TactileMap:
	def __init__(self, center, box_widths, granularity=10*np.ones(3), dim=3, w=np.array([1,1,0.5])):
		""" Occupancy Map generated using contact points. 
  			Based on Sequential Bayesian Hilbert Maps (SBHM) developed by Senanayake et al.
			paper: https://proceedings.mlr.press/v78/senanayake17a/senanayake17a.pdf 
			code: https://github.com/RansML/Bayesian_Hilbert_Maps/blob/master/BHM/original/sbhm.py#L82 
   
		Args:
			center (array): center point of the entire volume of occupancy map
			box_widths (array): occupancy map dimensions
			granularity (array, optional): resolution of each dimension. Defaults to 10*np.ones(3).
			dim (int, optional): occupancy map dimension. Defaults to 3.
			w (array, optional): weights used for the RBF kernel function. Defaults to np.array([1,1,0.5]).
		"""
		self.box_widths = box_widths
		self.center = center
		upper_limits = self.center + self.box_widths
		lower_limits = self.center - self.box_widths
		self.dim = dim
		
  		# indicator of whether having taken data or not
		self.has_data = False
  
		# create 2D/3D occupancy map based dimension
		if (self.dim == 3):
			x_range = np.arange(lower_limits[0], upper_limits[0], granularity[0])
			y_range = np.arange(lower_limits[1], upper_limits[1], granularity[1])
			z_range = np.arange(lower_limits[2], upper_limits[2], granularity[2])
			xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
			self.hinges = np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1), zz.ravel().reshape(-1,1)))
			self.w = w
		elif (self.dim == 2):
			xx, yy = np.meshgrid(np.linspace(lower_limits[0], upper_limits[0], granularity[0]),
								np.linspace(lower_limits[1], upper_limits[1], granularity[1]))
			self.hinges = np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)))
			self.w = w[:2]

	def fit_data(self, X2, y2, gamma, init_sigma=1):
		""" Run SHBM algorithm with positions in map and occupancy info

		Args:
			X2 (array): matrix of position in map already know occupancy state
			y2 (array): occupancy state (0/1)
			gamma (float): parameters for RBF kernel function
			init_sigma (int, optional): initial guess of variance. Defaults to 1.
		"""
		# X2 is points, y2 is occupancy indicator
		if (self.dim == 2):
			X2 = X2[:,:2]
		Phi = rbf_kernel(X2, self.hinges, gamma=gamma, w=self.w)

		N, D = Phi.shape[0], Phi.shape[1]
		print(N, D)
		xi = np.ones(N)
		self.mu = np.zeros(D)

		self.sig = init_sigma*np.ones(D) # initial sigma
		# run E and M step
		for i in range(5):
			# E-step
			self.mu, self.sig = self.calcPosterior(Phi, y2, xi, self.mu, self.sig)

			# M-step
			xi = np.sqrt(np.sum((Phi**2)*self.sig, axis=1) + (Phi.dot(self.mu.reshape(-1, 1))**2).ravel())

		self.has_data = True


	def predict_occ(self, input_pts, gamma, random_draws=1000):
		qPhi = rbf_kernel(input_pts, self.hinges, gamma=gamma, w=self.w)
		qw = np.random.multivariate_normal(self.mu, np.diag(self.sig), random_draws)
		#qw = np.random.multivariate_normal(mu, np.diag(sig), 50)
		occ = self.sigmoid(qw.dot(qPhi.T))
		occMean = np.mean(occ, axis=0)
		occStdev = np.std(occ, axis=0)
		return occMean, occStdev


	def visualize_map(self, gamma, granularity=10*np.ones(3), random_draws=1000, plot_mode='pyplot', threshold=0.52):
		""" Visualize Occupancy Map in 3D

		Args:
			gamma (float): parameter for RBF kernel function
			granularity (array, optional): resolution of occupancy map. Defaults to 10*np.ones(3).
			random_draws (int, optional): number of random draws. Defaults to 1000.
			plot_mode (str, optional): _description_. Defaults to 'pyplot'.
			threshold (float, optional): _description_. Defaults to 0.52.

		Returns:
			_type_: _description_
		"""
		self.upper_limits = self.center + self.box_widths
		self.lower_limits = self.center - self.box_widths
		if (self.dim == 3):
			x_range = np.arange(self.lower_limits[0], self.upper_limits[0], granularity[0])
			y_range = np.arange(self.lower_limits[1], self.upper_limits[1], granularity[1])
			z_range = np.arange(self.lower_limits[2], self.upper_limits[2], granularity[2])
			print(x_range.shape[0], y_range.shape[0], z_range.shape[0])
			xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
			qX = np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1), zz.ravel().reshape(-1,1)))
		elif (self.dim == 2):
			xx, yy = np.meshgrid(np.linspace(self.lower_limits[0], self.upper_limits[0], granularity[0]),
								np.linspace(self.lower_limits[1], self.upper_limits[1], granularity[1]))
			qX = np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)))
		import time
		time_stamp = time.time()
		qPhi = rbf_kernel(qX, self.hinges, gamma=gamma, w=self.w)
		print(f"qPhi took {time.time()-time_stamp} seconds")
		time_stamp = time.time()
		qw = np.random.multivariate_normal(self.mu, np.diag(self.sig), random_draws)
		print(f"qw took {time.time()-time_stamp} seconds")
		occ = self.sigmoid(qw.dot(qPhi.T))
		occMean = np.mean(occ, axis=0)
		occStdev = np.std(occ, axis=0)

		if (plot_mode == 'mcube'):
			# vertices, triangles = mcubes.marching_cubes_func(lower_limits, upper_limits, 100, 100, 100, f, 16)
			smoothed_shape = mcubes.smooth(occMean.reshape(xx.shape) < threshold)
			vertices, triangles = mcubes.marching_cubes(smoothed_shape, 0)
			mcubes.export_mesh(vertices, triangles, "mcube_output.dae", "MyShape")
		elif (plot_mode == 'plotly'):
			# Plot occupancy map
			fig = go.Figure()

			# map occMean to colormap seismic
			cmap = mpl.cm.get_cmap('hsv')

			# create RGBA vector from occMean
			occMeanRGBA = np.zeros((occMean.shape[0], 4))
			#occMeanRGBA[:,:3] = np.repeat(1-occMean[:,np.newaxis], 3, axis=1)
			# set occMeanRGBA to colormap of occMean
			occMeanRGBA[:,:3] = cmap(occMean)[:,:3]
			occMeanRGBA[:,3] = 0.6 
			# set alpha to zero for occMean < 0.5
			occMeanRGBA[occMean < threshold, 3] = 0
			if (self.dim == 3):

				fig.add_trace(go.Scatter3d(x=qX[:,0], y=qX[:,1],z=qX[:,2],
										mode='markers',
										marker=dict(
											# size=2*self.box_widths/granularity,
											size=4,
											color=occMeanRGBA,
											# colorscale='Bluered'
										),
										line=dict(color='rgba(0,0,0,0)'),
										name='lines'))

				fig.update_layout(title=f'3D reconstruction of map', 
								scene = dict(
									xaxis = dict(nticks=4, range=[self.lower_limits[0],self.upper_limits[0]],),
									yaxis = dict(nticks=4, range=[self.lower_limits[1],self.upper_limits[1]],),
									zaxis = dict(nticks=4, range=[self.lower_limits[2],self.upper_limits[2]],),
									aspectratio=dict(x=1, y=1, z=1)),
								xaxis_title='x axis (mm)', yaxis_title='y axis (mm)',
								autosize=False,
								width=800, height=600,
								scene_camera=dict(
												up=dict(x=0, y=0, z=1),
												center=dict(x=0, y=0, z=0),
												eye=dict(x=1, y=-2.1, z=1)),
								margin=dict(l=10, r=10, b=10, t=50))
			elif (self.dim == 2):
				fig.add_trace(go.Scatter(x=qX[:,0], y=qX[:,1],
										mode='markers',
										marker=dict(
											size=2*self.box_widths[0]/granularity[0],
											color=occMeanRGBA,
											colorscale='Bluered'
										),
										line=dict(color='rgba(0,0,0,0)'),
										name='lines'))
				fig.update_layout(title=f'2D reconstruction of map',
								xaxis_title='x axis (mm)', yaxis_title='y axis (mm)',
								autosize=False,
								width=1000, height=1000,
								margin=dict(l=10, r=10, b=10, t=50))

			return fig, qX, occMean
			# fig.show()
		elif (plot_mode == 'pyplot'):
			if (self.dim == 3):
				# map occMean to colormap seismic
				cmap = mpl.cm.get_cmap('Spectral')

				# create RGBA vector from occMean
				occMeanRGBA = np.zeros((occMean.shape[0], 4))
				#occMeanRGBA[:,:3] = np.repeat(1-occMean[:,np.newaxis], 3, axis=1)
				# set occMeanRGBA to colormap of occMean
				occMeanRGBA[:,:3] = cmap(occMean)[:,:3]
				occMeanRGBA[:,3] = 0.6 
				# set alpha to zero for occMean < 0.5
				occMeanRGBA[occMean < threshold, 3] = 0

				# 3d scatter plot of qX in pyplot
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(qX[:,0], qX[:,1], qX[:,2], c=occMeanRGBA, cmap='BuPu')
				ax.set_xlabel('x axis (mm)')
				ax.set_ylabel('y axis (mm)')
				ax.set_zlabel('z axis (mm)')
			elif (self.dim == 2):
				# map occMean to colormap seismic
				cmap = mpl.cm.get_cmap('Spectral')

				# create RGBA vector from occMean
				occMeanRGBA = np.zeros((occMean.shape[0], 4))
				#occMeanRGBA[:,:3] = np.repeat(1-occMean[:,np.newaxis], 3, axis=1)
				# set occMeanRGBA to colormap of occMean
				occMeanRGBA[:,:3] = cmap(occMean)[:,:3]
				occMeanRGBA[:,3] = 0.6
				# set alpha to zero for occMean < 0.5
				occMeanRGBA[occMean < 0.55, 3] = 0

				# 3d scatter plot of qX in pyplot
				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax.scatter(qX[:,0], qX[:,1], c=occMeanRGBA, cmap='seismic')
				ax.set_xlabel('x axis (mm)')
				ax.set_ylabel('y axis (mm)')

			return ax


	def sigmoid(self, x):
		return 1. / (1 + np.exp(-x))
		
	def calcPosterior(self, Phi, y, xi, mu0, sig0):
		logit_inv = self.sigmoid(xi)
		lam = 0.5 / xi * (logit_inv - 0.5)

		sig = 1. /(1./sig0 + 2*np.sum( (Phi.T**2)*lam, axis=1)) # note the numerical trick for the dot product

		mu = sig*(mu0/sig0 + np.dot(Phi.T, y - 0.5).ravel())

		return mu, sig

def process_occupancy_points(pts):
	return np.hstack((pts, np.ones((pts.shape[0],1)))) # add a column of ones to the end of the points

def sample_unoccupied_points_v2(body_pos, body_dims, num_samples, sensor_height=36, occ_gap=5):
	""" Sample unoccupied points in a specified volume of space

	Args:
		body_pos (_type_): _description_
		body_dims (_type_): _description_
		num_samples (_type_): _description_
		sensor_height (int, optional): _description_. Defaults to 36.
		occ_gap (int, optional): _description_. Defaults to 5.

	Returns:
		_type_: _description_
	"""
	# body_pos: 3xN array of body positions
	# body_dims: 3x2 array of body dimensions
	# num_samples: number of samples to take around each body

	unoccupied_body_points_all = np.zeros((0,3))
	for body_pt in body_pos:
		# sample unoccupied points around body_x, body_y, body_z
		unoccupied_points_x = np.random.uniform(low=body_pt[0]+body_dims[0,0], high=body_pt[0]+body_dims[0,1], size=(num_samples,1))
		unoccupied_points_y = np.random.uniform(low=body_pt[1]+body_dims[1,0], high=body_pt[1]+body_dims[1,1], size=(num_samples,1))
		unoccupied_points_z = np.random.uniform(low=body_pt[2]+body_dims[2,0], high=body_pt[2]+body_dims[2,1], size=(num_samples,1))
		# add points to unoccupied_points_all
		unoccupied_body_points_all = np.vstack((unoccupied_body_points_all,
										np.hstack((unoccupied_points_x,
													unoccupied_points_y,
													unoccupied_points_z))))

	return unoccupied_body_points_all

def sample_unoccupied_points(tracker, data, body_dims, num_samples, sensor_height=36, occ_gap=5):
	"""Sample points in a specified volume of space

	Args:
		tracker (_type_): _description_
		data (_type_): _description_
		body_dims (_type_): _description_
		num_samples (_type_): _description_
		sensor_height (int, optional): _description_. Defaults to 36.
		occ_gap (int, optional): _description_. Defaults to 5.

	Returns:
		_type_: _description_
	"""
	# sample points in the body frame
	# data: pandas dataframe
	# body_dims: 3x1 array
	# num_samples: int
	# returns: num_samples x 3 array of points in body frame

	# for every non-contact state, sample free space
	tracker.sensor_shape

	# loop through unique i values in data
	i_values = np.unique(data['i'])
	for i in i_values:
		# get data for each i value
		data_i = data[data['i'] == i]
		# sample unoccupied points from contact points
		contact_pts = data_i[['x','y','z']].to_numpy()
		unoccupied_points_all = np.zeros((0,3))
		# for contact_pt in contact_pts:
		# 	# sample points along the line of the sensor
		# 	unoccupied_points_x = np.ones((num_samples,1))*contact_pt[0]
		# 	unoccupied_points_y = np.random.uniform(low=contact_pt[1]-sensor_height, high=contact_pt[1]-occ_gap, size=(num_samples,1))
		# 	unoccupied_points_z = np.ones((num_samples,1))*contact_pt[2]
		# 	# add points to unoccupied_points_all
		# 	unoccupied_points_all = np.vstack((unoccupied_points_all, 
		# 								np.hstack((unoccupied_points_x, 
		# 											unoccupied_points_y, 
		# 											unoccupied_points_z))))

		body_pts = data_i[['body_x','body_y','body_z']].to_numpy()
		unoccupied_body_points_all = np.zeros((0,3))
		for body_pt in body_pts:
			# sample unoccupied points around body_x, body_y, body_z
			unoccupied_points_x = np.random.uniform(low=body_pt[0]-body_dims[0]/2, high=body_pt[0]+body_dims[0]/2, size=(num_samples,1))
			unoccupied_points_y = np.random.uniform(low=body_pt[1]-body_dims[1]/2, high=body_pt[1]+body_dims[1]/2, size=(num_samples,1))
			unoccupied_points_z = np.random.uniform(low=body_pt[2]-body_dims[2]/2, high=body_pt[2]+body_dims[2]/2, size=(num_samples,1))
			# add points to unoccupied_points_all
			unoccupied_body_points_all = np.vstack((unoccupied_body_points_all,
											np.hstack((unoccupied_points_x,
														unoccupied_points_y,
														unoccupied_points_z))))

		# create new dataframe with unoccupied points
		new_data = {'x': np.hstack((unoccupied_points_all[:,0], unoccupied_body_points_all[:,0])), 
	      			'y': np.hstack((unoccupied_points_all[:,1], unoccupied_body_points_all[:,1])),
					'z': np.hstack((unoccupied_points_all[:,2], unoccupied_body_points_all[:,2])),
					'occ': np.zeros(unoccupied_points_all.shape[0]+unoccupied_body_points_all.shape[0]),
					'i': np.hstack((np.ones(unoccupied_points_all.shape[0])*i, np.ones(unoccupied_body_points_all.shape[0])*i))}												

		# concatenate new_data to data dataframe
		data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)

	return data


def rbf_kernel(X, Y=None, gamma=None, w=None):
	"""Compute the rbf (gaussian) kernel between X and Y.

		K(x, y) = exp(-gamma ||x-y||^2)

	for each pair of rows x in X and y in Y.

	Read more in the :ref:`User Guide <rbf_kernel>`.

	Parameters
	----------
	X : ndarray of shape (n_samples_X, n_features)
		A feature array.

	Y : ndarray of shape (n_samples_Y, n_features), default=None
		An optional second feature array. If `None`, uses `Y=X`.

	gamma : float, default=None
		If None, defaults to 1.0 / n_features.

	Returns
	-------
	kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
		The RBF kernel.
	"""
	X, Y = check_pairwise_arrays(X, Y)
	if gamma is None:
		gamma = 1.0 / X.shape[1]
	K = euclidean_distances(X, Y, squared=True, w=w)
	K *= -gamma
	np.exp(K, K)  # exponentiate K in-place
	return K

def euclidean_distances( X, Y=None, *, Y_norm_squared=None, squared=False, X_norm_squared=None, w=None):
	"""
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        An array where each row is a sample and each column is a feature.
        If `None`, method uses `Y=X`.

    Y_norm_squared : array-like of shape (n_samples_Y,) or (n_samples_Y, 1) \
            or (1, n_samples_Y), default=None
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    squared : bool, default=False
        Return squared Euclidean distances.

    X_norm_squared : array-like of shape (n_samples_X,) or (n_samples_X, 1) \
            or (1, n_samples_X), default=None
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    """
	X, Y = check_pairwise_arrays(X, Y)

	if X_norm_squared is not None:
		X_norm_squared = check_array(X_norm_squared, ensure_2d=False)
		original_shape = X_norm_squared.shape
		if X_norm_squared.shape == (X.shape[0],):
			X_norm_squared = X_norm_squared.reshape(-1, 1)
		if X_norm_squared.shape == (1, X.shape[0]):
			X_norm_squared = X_norm_squared.T
		if X_norm_squared.shape != (X.shape[0], 1):
			raise ValueError(
				f"Incompatible dimensions for X of shape {X.shape} and "
				f"X_norm_squared of shape {original_shape}."
			)

	if Y_norm_squared is not None:
		Y_norm_squared = check_array(Y_norm_squared, ensure_2d=False)
		original_shape = Y_norm_squared.shape
		if Y_norm_squared.shape == (Y.shape[0],):
			Y_norm_squared = Y_norm_squared.reshape(1, -1)
		if Y_norm_squared.shape == (Y.shape[0], 1):
			Y_norm_squared = Y_norm_squared.T
		if Y_norm_squared.shape != (1, Y.shape[0]):
			raise ValueError(
				f"Incompatible dimensions for Y of shape {Y.shape} and "
				f"Y_norm_squared of shape {original_shape}."
			)

	return _euclidean_distances(X, Y, X_norm_squared, Y_norm_squared, squared, w=w)

def _euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False, w=None):
	"""Computational part of euclidean_distances

	Assumes inputs are already checked.

	If norms are passed as float32, they are unused. If arrays are passed as
	float32, norms needs to be recomputed on upcast chunks.
	TODO: use a float64 accumulator in row_norms to avoid the latter.
	"""
	# weighted norm
	if w is None:
		w = np.ones(X.shape[1], dtype=X.dtype)
	# w = np.array([1,1,0.5], dtype=np.float64)

	if X_norm_squared is not None:
		if X_norm_squared.dtype == np.float32:
			XX = None
		else:
			XX = X_norm_squared.reshape(-1, 1)
	elif X.dtype == np.float32:
		XX = None
	else:
		XX = row_norms(w*X, squared=True)[:, np.newaxis]

	if Y is X:
		YY = None if XX is None else XX.T
	else:
		if Y_norm_squared is not None:
			if Y_norm_squared.dtype == np.float32:
				YY = None
			else:
				YY = Y_norm_squared.reshape(1, -1)
		elif Y.dtype == np.float32:
			YY = None
		else:
			YY = row_norms(w*Y, squared=True)[np.newaxis, :]

	if X.dtype == np.float32:
		# To minimize precision issues with float32, we compute the distance
		# matrix on chunks of X and Y upcast to float64
		distances = _euclidean_distances_upcast(X, XX, Y, YY)
	else:
		# if dtype is already float64, no need to chunk and upcast
		distances = -2 * safe_sparse_dot(w*X, (w*Y).T, dense_output=True)
		distances += XX
		distances += YY

	# import ipdb;ipdb.set_trace()
	np.maximum(distances, 0, out=distances)

	# Ensure that distances between vectors and themselves are set to 0.0.
	# This may not be the case due to floating point rounding errors.
	if X is Y:
		np.fill_diagonal(distances, 0)

	return distances if squared else np.sqrt(distances, out=distances)



if __name__ == "__main__":

	import numpy as np
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	# from sklearn.metrics.pairwise import rbf_kernel

	# run tactileMap on cylinder_tracking.npz
	tm = TactileMap(center=np.array([0,-5,-20]), 
					box_widths=np.array([40,40,40])/2,
					granularity=13, dim=3)

	# load data into dataframe
	data = pd.read_csv('cone_tracking.csv')
	# load x, y, z into X2 array
	X2 = data[['x','y','z']].to_numpy()
	# load occ to y2 array
	y2 = data['occ'].to_numpy()

	# fit data
	tm.fit_data(X2=X2, y2=y2, gamma=0.2, init_sigma=1)

	# visualize map
	plot_mode = 'plotly'
	fig = tm.visualize_map(gamma=0.2, granularity=np.array([35,35,35]),
						random_draws=10,
						plot_mode=plot_mode,
						threshold=0.8)
	fig.show()