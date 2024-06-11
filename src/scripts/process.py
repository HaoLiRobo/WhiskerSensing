import GPy
import pandas as pd
import numpy as np
import torch
from scripts.model import DLSensorModel
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scripts.ukf_utils import wmean, wcov, unscentedTrans
from scripts.sensor import Sensor
from scipy.interpolate import splrep, BSpline

class ContactPointTracker:

    def __init__(self, fm_alpha=1.05,
                        fm_stride=1,
                        sigma_process=1e-5,
                        sigma_process_init=0.5,
                        sigma_sensor=0.01,
                        mu0=None,
                        tactile_map_function=None,
                        signal_threshold=5):
        """ UKF tracker for contact position estimation, take sequence of sensor signal and proprioception data and estimating
        position. The algorithm is the standard UKF as described in Table 3.4 in the book Probalistic Robotics on page 70.

        Args:
            fm_alpha (float, optional): fading memory parameter for scaling the covariance (1.01 to 1.1). Defaults to 1.05.
            fm_stride (int, optional): frequency that apply the scaling. Defaults to 1.
            sigma_process (_type_, optional): process noise used for covariance matrix in the predict step. Defaults to 1e-5.
            sigma_process_init (float, optional): initial guess of process noise used for covariance matrix in the predict step. Defaults to 0.5.
                                                  This value is important becasue it needs tuning for faster convergence at the beginnig
            sigma_sensor (float, optional): sensor noise used for covaraince matrix in update step. Defaults to 0.01.
            mu0 (_type_, optional): initial state estimate. Defaults to None.
            tactile_map_function (_type_, optional): deprecated for now. Defaults to None.
        """

        self.fm_alpha = fm_alpha            
        self.fm_stride = fm_stride          
        self.sigma_process = sigma_process               
        self.sigma_process_init = sigma_process_init    
        self.sigma_sensor = sigma_sensor
        
        self.nx = 3                     # state dimension size, x, y, z contact position
        self.sx = 2                     # observation dimension size, sensor signal
        if not mu0 is None:
            # initial state estimate
            self.mu0 = mu0              
        else:
            # if no initial guess, default to 0
            self.mu0 = np.zeros(self.nx)    
            
        self.fm_start_idx = 0                                   # starting step to apply the scaling
        self.fs = 200                                           # sampling speed (Hz)
        self.dt = 1.0/self.fs                                   # sampling period
        self.lam = 2                                            # scaling parameter that determine the distance from the mean point in UKF (gamma in Table 3.4)
        self.weight = np.zeros(2*self.nx+1)                     # weights vector for weighted sum of sigma points (step 4 in Table 3.4)
        self.weight[0] = self.lam/(self.nx + self.lam)
        self.weight[1:] = 1/(2 * (self.nx + self.lam))
        self.sigma0 = self.sigma_process_init*np.eye(self.nx)   # initial noise matrix (predict step)
        self.Q = self.sigma_process*np.eye(self.nx)             # process noise matrix (predict step)
        self.R = self.sigma_sensor*np.eye(self.sx)              # sensor noise matrix (update step)
        self.tactile_map_function = tactile_map_function        # ignore for now
        self.model = None
        self.signal_threshold = signal_threshold

    def init_calibration_model(self, calibration_fn, input_ref=[64.30451325,98.0687005,16.1023935], kernel=None, sensor_atten=50):
        """ Initialize the GPR model using collected calibration data. Obtain the model that maps contact position (x,y,z) to 
        sensor signal in x and y direction separately.

        Args:
            calibration_fn (string): calibration file name, which contains data of rig sweeping all over the sensing space
            input_ref (list, optional): the origin of the base of the whisker. Change this to what you measured at the beginning of calibration.
                                        substract this from calibration data will transform data to the sensor frame
            kernel (function, optional): kernel function of the GPR model. (RBF, Poly, ThinPlate, etc) 
                                        eg. ker1 = GPy.kern.RBF(3, lengthscale=5, variance=0.5)
                                            ker1 = GPy.kern.RBF(3, lengthscale=5, variance=5., ARD=True)
                                            ker1 = GPy.kern.Poly(3, variance=1.0, order=5, scale=0.1)
            sensor_atten (int, optional): scaling factor, scale down the raw sensor value to get better gradients, might need tuning. Defaults to 50.
        """
        data_tbl = pd.read_csv(calibration_fn)
        self.sensor_atten = sensor_atten
        
        P = data_tbl[['px','py','pz']].to_numpy() - input_ref   # contact position in the sensor frame
        S = data_tbl[['sx','sy']].to_numpy()/self.sensor_atten  # sensor signal after scaling
        self.mx = GPy.models.GPRegression(P,S[:,0:1],kernel)    # sensor model in x direction
        self.my = GPy.models.GPRegression(P,S[:,1:2],kernel)    # sensor model in y direction
    def init_optic_calibration_model(self, calibration_fn, input_ref=[64.30451325,98.0687005,16.1023935], kernel=None, sensor_atten=50):
        """ Initialize the GPR model using collected calibration data. Obtain the model that maps contact position (x,y,z) to 
        sensor signal in x and y direction separately.

        Args:
            calibration_fn (string): calibration file name, which contains data of rig sweeping all over the sensing space
            input_ref (list, optional): the origin of the base of the whisker. Change this to what you measured at the beginning of calibration.
                                        substract this from calibration data will transform data to the sensor frame
            kernel (function, optional): kernel function of the GPR model. (RBF, Poly, ThinPlate, etc) 
                                        eg. ker1 = GPy.kern.RBF(3, lengthscale=5, variance=0.5)
                                            ker1 = GPy.kern.RBF(3, lengthscale=5, variance=5., ARD=True)
                                            ker1 = GPy.kern.Poly(3, variance=1.0, order=5, scale=0.1)
            sensor_atten (int, optional): scaling factor, scale down the raw sensor value to get better gradients, might need tuning. Defaults to 50.
        """
        data_tbl = pd.read_csv(calibration_fn)
        self.sensor_atten = sensor_atten
        
        P = data_tbl[['px','py','pz']].to_numpy() - input_ref   # contact position in the sensor frame
        
        S = data_tbl[['sx','sy']].to_numpy()  # sensor signal after scaling
        S[:,0] = (S[:,0]- S[0,0]) / sensor_atten
        S[:,1] = (S[:,1]- S[0,1]) / sensor_atten
        self.mx = GPy.models.GPRegression(P,S[:,0:1],kernel)    # sensor model in x direction
        self.my = GPy.models.GPRegression(P,S[:,1:2],kernel)    # sensor model in y direction
    
    def init_DL_model(self, cfg_dict):
        self.device = torch.device("mps")
        self.sensor_atten = cfg_dict['sensor_atten']
        
        # load calibration data collected for the testing sensor
        calib_data = pd.read_csv(cfg_dict['calib_pth'])
        S = calib_data[['sx','sy']].to_numpy()
        self.max_s = S.max()
        self.min_s = S.min()
        
        # load sensor shape matrix
        shape_x = pd.read_csv(cfg_dict['shape_x_pth'])
        pts_x = shape_x[['px']].to_numpy()[:,0]
        pts_z = shape_x[['pz']].to_numpy()[:,0]
        shape_y = pd.read_csv(cfg_dict['shape_y_pth'])
        pts_x_new = shape_y[['px']].to_numpy()[:,0]
        pts_y = shape_y[['py']].to_numpy()[:,0]
        m_shape = BSpline(*splrep(x=pts_x[::-1], y=pts_z[::-1], s=2))
        X = pts_x_new - pts_x_new[0]
        Y = pts_y - pts_y[0]
        Z = m_shape(pts_x_new) - m_shape(pts_x_new)[0]
        self.shape_numpy = np.vstack((X, Y, Z)).T
        
        # load model
        self.model = DLSensorModel(cfg_dict['pos_ch'], cfg_dict['shape_ch'])
        self.model.load_state_dict(torch.load(cfg_dict['model_pth']))
        self.model.to(self.device)
        self.model.eval()

    def init_sensor_model(self, sensor_num, sensor_atten=50, kernel=None):
        """ Another way to initialize GPR model. Load data from npz files and initialize Sensor instance with GPR model.

        Args:
            sensor_num (_type_): the sensor No.
            sensor_atten (int, optional): scaling factor, scale down the raw sensor value to get better gradients, might need tuning. Defaults to 50.
            kernel (_type_, optional): kernel function to use.
        """
        self.sensor_atten = sensor_atten

        # initialize sensor instance, which load data from npz file and fit GPR model with selected kernel function
        self.sensor = Sensor(sensor_num, kernel=kernel)
        self.sensor_shape = self.sensor.sensor_shape
        self.mx = self.sensor.mx
        self.my = self.sensor.my
        
    def initialize_compare(self, track_data_fn, input_ref=None, offset_mode='sensor', sensor_atten=50, signal_threshold=5):
        """ Initialize the tracker with collected data from a csv file.

        Args:
            track_data_fn (string): csv file that stores one sweeping trial to be tracked, contains proprioception data and raw sensor reading
            input_ref (list, optional): origin of the based of the whisker.
            offset_mode (str, optional): can be 'sensor' or 'from_data'.
                                        - sensor mode: sensor fixed and the calbiration rig moving with the stage (for testing data)
                                        - from_data mode: sensor moving on the stage and object fixed in the table 
            sensor_atten (int, optional): scaling down the raw sensor data (from ~1000 of raw value) (This is due to different sensor having different sensing range).
                                          scale down the raw sensor value to get better gradients, might need tuning. Defaults to 50.
        """
        
        data_tbl = pd.read_csv(track_data_fn)
        
        max_px = data_tbl.px.to_numpy().max()
        
        # load stage positions (proprioception data)
        state = np.vstack((data_tbl.px.to_numpy(),
                                data_tbl.py.to_numpy(),
                                data_tbl.pz.to_numpy())).T

        # transform data to some frame based on how data is collected
        # this is aimed to select an origin for the estimated contact position
        # the algorithm will still work without this transforming 
        # but it will be difficult to align the contacts with CAD models
        if (offset_mode == 'sensor'):
            # sensor is fixed on stage platform, the origin reference is the origin of the whisker base
            # having this origin will make all the estimated contact positions lie in sensor frame
            coord_ref = input_ref
        elif (offset_mode == 'from_data'):
            # sensor is moving with the motors, the origin reference is a point on the contact object
            # having this origin will make all the estimated contact posotions lie in the same frame
            # with CAD models, convenient to calculate contact errors.
            # this orgin is obtained by let the left bottom hex standoff tip making contact with the 
            # selected point on the contact object, then measuring the relative position from sensor 
            # base to that contact position.
            # standoff2sensor_offset = np.array([9.703, 5.5, 0])      # relative x and y position from sensor base to the hex standoff
            # hole2standoff_offset = np.array([70.1297, 92.3684, 0])  # x and y position of the selected point in world frame
            # z_offset = np.array([0, 0,  47.02211512])               # total relative z offset of the sensor base
            # coord_ref = hole2standoff_offset - standoff2sensor_offset + z_offset
            
            corner2sensor_offset = np.array([-3.91, 41.91, 15.43])      # relative x and y position from sensor base to the hex standoff
            # corner2sensor_offset = np.array([-3.91, 41.91, 15.43])      # relative x and y position from sensor base to the hex standoff
            tapeinter2corner_offset = np.array([45.379814625 - 2.71 + 74.47,93.002338125 + 24.53 + 3,48.490870125])  # x and y position of the selected point in world frame
            coord_ref = tapeinter2corner_offset - corner2sensor_offset

        # stage positions are then transformed into the selected frame
        self.coord_ref = coord_ref
        state -= coord_ref
        
        # load raw sensor signal
        sensor_signal = np.vstack((data_tbl.sx.to_numpy(),
                                data_tbl.sy.to_numpy(),
                                data_tbl.sz.to_numpy())).T
        
        # apply the scaling

        sensor_signal = sensor_signal[:,:2]/sensor_atten
        self.signal_threshold = signal_threshold
                            
        self.X_t = state                    # stage positions (proprioception data)
        self.Y_t = sensor_signal            # raw sensor signal
        self.ep_len = self.Y_t.shape[0]     # episode length of the sweeping process
        nx = self.nx                        # contact position dimension -> 3
        sx = self.sx                        # sensor signal dimension -> 2

        # eqn 5.8 in Thesis
        # process proprio state into change of stage position 
        # use to update contact poistion (adding delta_t * vel)
        self.dX_t = np.zeros_like(self.X_t)
        self.dX_t[1:,:] = self.X_t[1:,:]-self.X_t[:-1,:]

        # the contact position (mean of the distribution) (3 * length of ep)
        self.x_ukf = np.zeros([nx,self.ep_len])
        # set initial mean estimate
        self.x_ukf[:,0] = self.mu0      
        # variance of the distribution (3*3*length) (Sigma matrix in the algo)
        self.sig_ukf = np.zeros([nx,nx,self.ep_len])
        # set initial variance, noise of uniform distribution
        self.sig_ukf[:,:,0] = self.sigma0                
        # square root of sigma matrix
        self.sig_sqrtm = np.zeros([nx,nx,self.ep_len])
        self.sig_sqrtm[:,:,0] = sqrtm(self.sigma0)
        # vector for applying the fading memory scalar for covariance
        self.sigma_scalar = np.ones([1,self.ep_len])
        # apply it at steps based on starting index and stride
        self.sigma_scalar[0, self.fm_start_idx:-1:self.fm_stride] = self.fm_alpha
        # estimated sensor reading based on current mean estimate and sensor model
        self.y_est = np.zeros([sx, self.ep_len])
        # convariance matrix for sensor signal
        self.cov_yy = np.zeros([sx, sx, self.ep_len])
        # covariance matrix between sensor signal and contact position
        self.cov_xy = np.zeros([nx, sx, self.ep_len])
        
    def initialize(self, track_data_fn, input_ref=None, offset_mode='sensor', sensor_atten=50, signal_threshold=5):
        """ Initialize the tracker with collected data from a csv file.

        Args:
            track_data_fn (string): csv file that stores one sweeping trial to be tracked, contains proprioception data and raw sensor reading
            input_ref (list, optional): origin of the based of the whisker.
            offset_mode (str, optional): can be 'sensor' or 'from_data'.
                                        - sensor mode: sensor fixed and the calbiration rig moving with the stage (for testing data)
                                        - from_data mode: sensor moving on the stage and object fixed in the table 
            sensor_atten (int, optional): scaling down the raw sensor data (from ~1000 of raw value) (This is due to different sensor having different sensing range).
                                          scale down the raw sensor value to get better gradients, might need tuning. Defaults to 50.
        """
        
        data_tbl = pd.read_csv(track_data_fn)
        
        # load stage positions (proprioception data)
        state = np.vstack((data_tbl.px.to_numpy(),
                                data_tbl.py.to_numpy(),
                                data_tbl.pz.to_numpy())).T

        # transform data to some frame based on how data is collected
        # this is aimed to select an origin for the estimated contact position
        # the algorithm will still work without this transforming 
        # but it will be difficult to align the contacts with CAD models
        if (offset_mode == 'sensor'):
            # sensor is fixed on stage platform, the origin reference is the origin of the whisker base
            # having this origin will make all the estimated contact positions lie in sensor frame
            coord_ref = input_ref
        elif (offset_mode == 'from_data'):
            # sensor is moving with the motors, the origin reference is a point on the contact object
            # having this origin will make all the estimated contact posotions lie in the same frame
            # with CAD models, convenient to calculate contact errors.
            # this orgin is obtained by let the left bottom hex standoff tip making contact with the 
            # selected point on the contact object, then measuring the relative position from sensor 
            # base to that contact position.
            standoff2sensor_offset = np.array([9.703, 5.5, 0])      # relative x and y position from sensor base to the hex standoff
            hole2standoff_offset = np.array([70.1297, 92.3684, 0])  # x and y position of the selected point in world frame
            z_offset = np.array([0, 0,  47.02211512])               # total relative z offset of the sensor base
            coord_ref = hole2standoff_offset - standoff2sensor_offset + z_offset

        # stage positions are then transformed into the selected frame
        self.coord_ref = coord_ref
        state -= coord_ref
        
        # load raw sensor signal
        sensor_signal = np.vstack((data_tbl.sx.to_numpy(),
                                data_tbl.sy.to_numpy(),
                                data_tbl.sz.to_numpy())).T
        
        # apply the scaling
        sensor_signal [:,0]= sensor_signal[:,0] - sensor_signal[0,0]
        sensor_signal [:,1] = sensor_signal[:,1] - sensor_signal[0,1]
        sensor_signal = sensor_signal[:,:2]/sensor_atten
        print(sensor_signal)
        self.signal_threshold = signal_threshold
                            
        self.X_t = state                    # stage positions (proprioception data)
        self.Y_t = sensor_signal            # raw sensor signal
        self.ep_len = self.Y_t.shape[0]     # episode length of the sweeping process
        nx = self.nx                        # contact position dimension -> 3
        sx = self.sx                        # sensor signal dimension -> 2

        # eqn 5.8 in Thesis
        # process proprio state into change of stage position 
        # use to update contact poistion (adding delta_t * vel)
        self.dX_t = np.zeros_like(self.X_t)
        self.dX_t[1:,:] = self.X_t[1:,:]-self.X_t[:-1,:]

        # the contact position (mean of the distribution) (3 * length of ep)
        self.x_ukf = np.zeros([nx,self.ep_len])
        # set initial mean estimate
        self.x_ukf[:,0] = self.mu0      
        # variance of the distribution (3*3*length) (Sigma matrix in the algo)
        self.sig_ukf = np.zeros([nx,nx,self.ep_len])
        # set initial variance, noise of uniform distribution
        self.sig_ukf[:,:,0] = self.sigma0                
        # square root of sigma matrix
        self.sig_sqrtm = np.zeros([nx,nx,self.ep_len])
        self.sig_sqrtm[:,:,0] = sqrtm(self.sigma0)
        # vector for applying the fading memory scalar for covariance
        self.sigma_scalar = np.ones([1,self.ep_len])
        # apply it at steps based on starting index and stride
        self.sigma_scalar[0, self.fm_start_idx:-1:self.fm_stride] = self.fm_alpha
        # estimated sensor reading based on current mean estimate and sensor model
        self.y_est = np.zeros([sx, self.ep_len])
        # convariance matrix for sensor signal
        self.cov_yy = np.zeros([sx, sx, self.ep_len])
        # covariance matrix between sensor signal and contact position
        self.cov_xy = np.zeros([nx, sx, self.ep_len])
    
    def initialize_with_gaussian_noise(self, track_data_fn, input_ref=None, offset_mode='sensor', sensor_atten=50, mean=0, var=1.0):
        """ Initialize the tracker with collected data from a csv file.

        Args:
            track_data_fn (string): csv file that stores one sweeping trial to be tracked, contains proprioception data and raw sensor reading
            input_ref (list, optional): origin of the based of the whisker.
            offset_mode (str, optional): can be 'sensor' or 'from_data'.
                                        - sensor mode: sensor fixed and the calbiration rig moving with the stage (for testing data)
                                        - from_data mode: sensor moving on the stage and object fixed in the table 
            sensor_atten (int, optional): scaling down the raw sensor data (from ~1000 of raw value) (This is due to different sensor having different sensing range).
                                          scale down the raw sensor value to get better gradients, might need tuning. Defaults to 50.
        """
        
        data_tbl = pd.read_csv(track_data_fn)
        
        # load stage positions (proprioception data)
        state = np.vstack((data_tbl.px.to_numpy(),
                                data_tbl.py.to_numpy(),
                                data_tbl.pz.to_numpy())).T

        # transform data to some frame based on how data is collected
        # this is aimed to select an origin for the estimated contact position
        # the algorithm will still work without this transforming 
        # but it will be difficult to align the contacts with CAD models
        if (offset_mode == 'sensor'):
            # sensor is fixed on stage platform, the origin reference is the origin of the whisker base
            # having this origin will make all the estimated contact positions lie in sensor frame
            coord_ref = input_ref
        elif (offset_mode == 'from_data'):
            # sensor is moving with the motors, the origin reference is a point on the contact object
            # having this origin will make all the estimated contact posotions lie in the same frame
            # with CAD models, convenient to calculate contact errors.
            # this orgin is obtained by let the left bottom hex standoff tip making contact with the 
            # selected point on the contact object, then measuring the relative position from sensor 
            # base to that contact position.
            standoff2sensor_offset = np.array([9.703, 5.5, 0])      # relative x and y position from sensor base to the hex standoff
            hole2standoff_offset = np.array([70.1297, 92.3684, 0])  # x and y position of the selected point in world frame
            z_offset = np.array([0, 0,  47.02211512])               # total relative z offset of the sensor base
            coord_ref = hole2standoff_offset - standoff2sensor_offset + z_offset

        # stage positions are then transformed into the selected frame
        self.coord_ref = coord_ref
        state -= coord_ref
        
        # load raw sensor signal
        sensor_signal = np.vstack((data_tbl.sx.to_numpy(),
                                data_tbl.sy.to_numpy(),
                                data_tbl.sz.to_numpy())).T
        
        # apply the scaling
        sensor_signal = sensor_signal[:,:2]/sensor_atten
                            
        self.X_t = state                    # stage positions (proprioception data)
        self.Y_t = sensor_signal            # raw sensor signal
        self.ep_len = self.Y_t.shape[0]     # episode length of the sweeping process
        nx = self.nx                        # contact position dimension -> 3
        sx = self.sx                        # sensor signal dimension -> 2

        # eqn 5.8 in Thesis
        # process proprio state into change of stage position 
        # use to update contact poistion (adding delta_t * vel)
        self.dX_t = np.zeros_like(self.X_t)
        self.dX_t[1:,:] = self.X_t[1:,:]-self.X_t[:-1,:]
        
        # Add noise to velocity
        noise = np.random.normal(mean, var, size=self.dX_t.shape)
        self.dX_t += noise * 1 / 200

        # the contact position (mean of the distribution) (3 * length of ep)
        self.x_ukf = np.zeros([nx,self.ep_len])
        # set initial mean estimate
        self.x_ukf[:,0] = self.mu0      
        # variance of the distribution (3*3*length) (Sigma matrix in the algo)
        self.sig_ukf = np.zeros([nx,nx,self.ep_len])
        # set initial variance, noise of uniform distribution
        self.sig_ukf[:,:,0] = self.sigma0                
        # square root of sigma matrix
        self.sig_sqrtm = np.zeros([nx,nx,self.ep_len])
        self.sig_sqrtm[:,:,0] = sqrtm(self.sigma0)
        # vector for applying the fading memory scalar for covariance
        self.sigma_scalar = np.ones([1,self.ep_len])
        # apply it at steps based on starting index and stride
        self.sigma_scalar[0, self.fm_start_idx:-1:self.fm_stride] = self.fm_alpha
        # estimated sensor reading based on current mean estimate and sensor model
        self.y_est = np.zeros([sx, self.ep_len])
        # convariance matrix for sensor signal
        self.cov_yy = np.zeros([sx, sx, self.ep_len])
        # covariance matrix between sensor signal and contact position
        self.cov_xy = np.zeros([nx, sx, self.ep_len])
        
    def initialize_with_IMU_noise_model(self, track_data_fn, input_ref=None, offset_mode='sensor', sensor_atten=50, sigma_a=0, sigma_ba=1.0, sample_rate=200, aggregation = False):
        """ Initialize the tracker with collected data from a csv file.

        Args:
            track_data_fn (string): csv file that stores one sweeping trial to be tracked, contains proprioception data and raw sensor reading
            input_ref (list, optional): origin of the based of the whisker.
            offset_mode (str, optional): can be 'sensor' or 'from_data'.
                                        - sensor mode: sensor fixed and the calbiration rig moving with the stage (for testing data)
                                        - from_data mode: sensor moving on the stage and object fixed in the table 
            sensor_atten (int, optional): scaling down the raw sensor data (from ~1000 of raw value) (This is due to different sensor having different sensing range).
                                          scale down the raw sensor value to get better gradients, might need tuning. Defaults to 50.
        """
        
        data_tbl = pd.read_csv(track_data_fn)
        
        # load stage positions (proprioception data)
        state = np.vstack((data_tbl.px.to_numpy(),
                                data_tbl.py.to_numpy(),
                                data_tbl.pz.to_numpy())).T

        # transform data to some frame based on how data is collected
        # this is aimed to select an origin for the estimated contact position
        # the algorithm will still work without this transforming 
        # but it will be difficult to align the contacts with CAD models
        if (offset_mode == 'sensor'):
            # sensor is fixed on stage platform, the origin reference is the origin of the whisker base
            # having this origin will make all the estimated contact positions lie in sensor frame
            coord_ref = input_ref
        elif (offset_mode == 'from_data'):
            # sensor is moving with the motors, the origin reference is a point on the contact object
            # having this origin will make all the estimated contact posotions lie in the same frame
            # with CAD models, convenient to calculate contact errors.
            # this orgin is obtained by let the left bottom hex standoff tip making contact with the 
            # selected point on the contact object, then measuring the relative position from sensor 
            # base to that contact position.
            standoff2sensor_offset = np.array([9.703, 5.5, 0])      # relative x and y position from sensor base to the hex standoff
            hole2standoff_offset = np.array([70.1297, 92.3684, 0])  # x and y position of the selected point in world frame
            z_offset = np.array([0, 0,  47.02211512])               # total relative z offset of the sensor base
            coord_ref = hole2standoff_offset - standoff2sensor_offset + z_offset

        # stage positions are then transformed into the selected frame
        self.coord_ref = coord_ref
        state -= coord_ref
        
        # load raw sensor signal
        sensor_signal = np.vstack((data_tbl.sx.to_numpy(),
                                data_tbl.sy.to_numpy(),
                                data_tbl.sz.to_numpy())).T
        
        # apply the scaling
        sensor_signal = sensor_signal[:,:2]/sensor_atten
                            
        self.X_t = state                    # stage positions (proprioception data)
        self.Y_t = sensor_signal            # raw sensor signal
        self.ep_len = self.Y_t.shape[0]     # episode length of the sweeping process
        nx = self.nx                        # contact position dimension -> 3
        sx = self.sx                        # sensor signal dimension -> 2

        # eqn 5.8 in Thesis
        # process proprio state into change of stage position 
        # use to update contact poistion (adding delta_t * vel)
        self.dX_t = np.zeros_like(self.X_t)
        self.dX_t[1:,:] = self.X_t[1:,:]-self.X_t[:-1,:]
        
        # Add IMU noise
        def add_IMU_noise(position_array, sample_rate, noise_density, random_walk, aggregation):
            """
            Integrate 3D acceleration to estimate change in position, including simulated noise and random walk.
            """
            positions = position_array  # Initial positions from self.X_t
            delta_t = 1 / sample_rate
            velocities = np.zeros((len(positions), 3))  # Initial velocities are assumed to be zero
            accelerations = np.diff(positions, n=2, axis=0, prepend=np.zeros((2,3))) / delta_t**2 
            # considering the case that the sensor has already been working for some time （30 mins)
            if aggregation:
                # addting initial aggregated error to velocity: random_walk * sqrt(operating_time)
                velocities[0,:] = random_walk * np.sqrt(30*60)
                # adding initial aggregated error to acceleration: bias_instability * sqrt(operating_time)
                accelerations += 0.001 * np.sqrt(30*60)
            position_changes = [velocities[0,:]]

            for i in range(1, len(positions)):

                # Simulate accelerometer noise and bias for each axis (x, y, z)
                noise = noise_density * np.sqrt(1 / delta_t) * np.random.normal(0, 1, 3)
                bias = random_walk * np.sqrt(delta_t) * np.random.normal(0, 1, 3)

                # Apply noise and bias to acceleration
                noisy_acceleration = accelerations[i - 1] + noise + bias

                velocities[i] = velocities[i - 1] + noisy_acceleration * delta_t
                positions[i] = positions[i - 1] + velocities[i] * delta_t + 0.5 * noisy_acceleration * delta_t ** 2
                # Calculate the change in position from the previous step
                position_changes.append(positions[i] - positions[i - 1])

            return np.array(position_changes)
        self.dX_t = add_IMU_noise(np.copy(self.X_t), sample_rate, sigma_a * 1000, sigma_ba*1000, aggregation=aggregation)
        
        # delta_t = 1 / sample_rate
        # # white noise
        # sigma_ad = sigma_a * 1 / np.sqrt(sample_rate) * 1000 # m to mm
        # white_noise = sigma_ad * np.random.normal(0, 1, size=self.dX_t.shape)
        # # bias
        # bias = np.zeros_like(self.dX_t)
        # sigma_bad = sigma_ba * np.sqrt(sample_rate) * 1000 # m to mm
        # for i in range(1,len(bias)):
        #     bias[i] = bias[i-1] + sigma_bad * np.random.normal(0, 1, size=self.dX_t.shape[1])
        # acc = white_noise + bias
        # # integrate the acc and add it to the robot states
        # self.dX_t += delta_t * 0.5 * acc * delta_t

        # the contact position (mean of the distribution) (3 * length of ep)
        self.x_ukf = np.zeros([nx,self.ep_len])
        # set initial mean estimate
        self.x_ukf[:,0] = self.mu0      
        # variance of the distribution (3*3*length) (Sigma matrix in the algo)
        self.sig_ukf = np.zeros([nx,nx,self.ep_len])
        # set initial variance, noise of uniform distribution
        self.sig_ukf[:,:,0] = self.sigma0                
        # square root of sigma matrix
        self.sig_sqrtm = np.zeros([nx,nx,self.ep_len])
        self.sig_sqrtm[:,:,0] = sqrtm(self.sigma0)
        # vector for applying the fading memory scalar for covariance
        self.sigma_scalar = np.ones([1,self.ep_len])
        # apply it at steps based on starting index and stride
        self.sigma_scalar[0, self.fm_start_idx:-1:self.fm_stride] = self.fm_alpha
        # estimated sensor reading based on current mean estimate and sensor model
        self.y_est = np.zeros([sx, self.ep_len])
        # convariance matrix for sensor signal
        self.cov_yy = np.zeros([sx, sx, self.ep_len])
        # covariance matrix between sensor signal and contact position
        self.cov_xy = np.zeros([nx, sx, self.ep_len])

    def initialize_from_data(self, pos_data, sensor_data, input_ref=None, offset_mode='sensor', sensor_atten=50):
        """ Initialization directly from input data. 

        Args:
            pos_data (array): stage position of entire sweeping process
            sensor_data (array): sensor signal of entire sweeping process
            input_ref (list, optional): origin of the based of the whisker.
            offset_mode (str, optional): can be 'sensor' or 'from_data'.
                                        - sensor mode: sensor fixed and the calbiration rig moving with the stage (for testing data)
                                        - from_data mode: sensor moving on the stage and object fixed in the table 
            sensor_atten (int, optional): scaling down the raw sensor data (from ~1000 of raw value) (This is due to different sensor having different sensing range).
                                          scale down the raw sensor value to get better gradients, might need tuning. Defaults to 50.
        """
        # load data from input, check shape
        state = pos_data
        sensor_signal = sensor_data
        assert len(sensor_signal.shape) == 2, "sensor signal shape wrong"
        assert sensor_signal.shape[1] == 3, "sensor signal shape wrong"

        # transform data to some frame based on how data is collected
        # this is aimed to select an origin for the estimated contact position
        # the algorithm will still work without this transforming 
        # but it will be difficult to align the contacts with CAD models
        if (offset_mode == 'sensor'):
            # sensor is fixed on stage platform, the origin reference is the origin of the whisker base
            # having this origin will make all the estimated contact positions lie in sensor frame
            # (the state is the ground truth contact position in sensor frame)
            coord_ref = input_ref
        elif (offset_mode == 'from_data'):
            # sensor is moving with the motors, the origin reference is a point on the contact object
            # having this origin will make all the estimated contact posotions lie in the same frame
            # with CAD models, convenient to calculate contact errors.
            # this orgin is obtained by let the right bottom hex standoff tip making contact with the 
            # selected point on the contact object, then measuring the relative position from sensor 
            # base to that contact position.
            standoff2sensor_offset = np.array([9.703, 5.5, 0])      # relative x and y position from sensor base to the hex standoff
            hole2standoff_offset = np.array([70.1297, 92.3684, 0])  # x and y position of the selected point in world frame
            z_offset = np.array([0, 0,  47.02211512])               # total relative z offset of the sensor base
            coord_ref = hole2standoff_offset - standoff2sensor_offset + z_offset
            
        # stage positions are then transformed into the selected frame
        self.coord_ref = coord_ref
        state -= coord_ref
        
        # load raw signal data from input and apply scaling
        sensor_signal = sensor_signal[:,:2]/sensor_atten
                            
        self.X_t = state                    # stage positions (proprioception data)
        self.Y_t = sensor_signal            # raw sensor signal
        self.ep_len = self.Y_t.shape[0]     # episode length of the sweeping process
        nx = self.nx                        # contact position dimension -> 3
        sx = self.sx                        # sensor signal dimension -> 2

        # process proprio state into change of position
        self.dX_t = np.zeros_like(self.X_t)
        self.dX_t[1:,:] = self.X_t[1:,:]-self.X_t[:-1,:]

        # the contact position (mean of the distribution) (3 * length of ep)
        self.x_ukf = np.zeros([nx,self.ep_len])
        # set initial mean estimate
        self.x_ukf[:,0] = self.mu0      
        # variance of the distribution (3*3*length) (Sigma matrix in the algo)
        self.sig_ukf = np.zeros([nx,nx,self.ep_len])
        # set initial variance, noise of uniform distribution
        self.sig_ukf[:,:,0] = self.sigma0                
        # square root of sigma matrix
        self.sig_sqrtm = np.zeros([nx,nx,self.ep_len])
        self.sig_sqrtm[:,:,0] = sqrtm(self.sigma0)
        # vector for applying the fading memory scalar for covariance
        self.sigma_scalar = np.ones([1,self.ep_len])
        # apply it at steps based on starting index and stride
        self.sigma_scalar[0, self.fm_start_idx:-1:self.fm_stride] = self.fm_alpha
        # estimated sensor reading based on current mean estimate and sensor model
        self.y_est = np.zeros([sx, self.ep_len])
        # convariance matrix for sensor signal
        self.cov_yy = np.zeros([sx, sx, self.ep_len])
        # covariance matrix between sensor signal and contact position
        self.cov_xy = np.zeros([nx, sx, self.ep_len])
        
    def run_tracker(self, use_map=False, track_mode='world', init_pos_offset=np.zeros(3)):
        """ Run the UKF algorithm to estimate the contact positions
        
        Args:
            use_map: deprecated for now.
            track_mode: mode selection for mean estimation.
                        - 'world': rig moving on the stage and sensor fixed, setting mean 
                        estimated fixed known offset from the ground truth contact position
                        - 'static': sensor moving and contact object fixed, always guess 
                        initial position some where in the sensor frame
            init_pos_offset: some offset added to the ground truth contact position as initial guess
                        to test how the algorithm works.
        """
        # tracking flag, indicating trakcing status
        # flipped when sensor signal is below/above some threshold
        started_tracking = False
        # use the initial sensor value as bias to infer signal strength
        sensor_bias = self.Y_t[0,:]
        # timestep index
        ti = 0
        # for ploting, get rid of no contact state
        selection_arr = []
        # index that indicate the current timestep belongs to which sweeping trial
        segment_idx = 0

        Xnew_save = np.zeros([self.nx,2*self.nx+1,self.ep_len])

        for i in range(1,self.ep_len):
            ti += 1
            # calculated current signal strength
            signal_strength = np.linalg.norm(self.Y_t[i] - sensor_bias)
            # trakcing in/not contact
            if signal_strength < self.signal_threshold:
                selection_arr.append(0)
                if (started_tracking):
                    # flip flag due to lost of tracking
                    started_tracking = False
                else:
                    # keep flag if no signal
                    # keep previous estimation of mean and variance
                    self.x_ukf[:, ti] = self.x_ukf[:, ti-1]
                    self.sig_ukf[:, :, ti] = self.sig_ukf[:, :, ti-1]
                continue
            else:
                if (not started_tracking):
                    # initialize contact position estimation if tracking starts
                    # use map on shape to get most likely point, ignore use_map for now
                    if (use_map):
                        init_loc_idx = self.tactile_map_function(self.X_t[ti-1,:])
                        if init_loc_idx != -1:
                            self.mu0 = self.sensor.sensor_shape[init_loc_idx,:]

                    # for visualization purpose
                    # start a new trial if start a new tracking
                    # state changes from no contact to contact
                    segment_idx += 1
                    if (track_mode == 'world'):
                        # rig moving on the stage and sensor fixed
                        # setting mean estimated fixed known offset from the ground truth contact position
                        self.x_ukf[:, ti-1] = self.X_t[ti-1,:] + init_pos_offset
                    elif (track_mode == 'static'):
                        # always guess initial position some where in the sensor frame
                        self.x_ukf[:, ti-1] = self.mu0
                    # initialize covariance
                    self.sig_ukf[:, :, ti-1] = self.sigma0
                selection_arr.append(segment_idx)
                started_tracking = True

            # perform sampling of unscented points
            X = unscentedTrans(self.lam, self.x_ukf[:,ti - 1], self.sig_ukf[:, :, ti - 1])

            """ Predict Step """
            # predict using process model x_new = x_prev + delta_t * vel (1/200 * vel)
            Xnew = X + np.expand_dims(self.dX_t[ti,:], 1)
            # save the new unscented points in a matrix
            Xnew_save[:,:,ti] = Xnew

            # predict mean and covariance
            # perform step 4 in Table 3.4, weighted sum of all unscented points
            self.x_ukf[:, ti] = wmean(self.weight, Xnew)
            # perform step 5 in Table 3.4
            self.sig_ukf[:, :, ti] = wcov(self.weight, Xnew, Xnew) + self.Q

            # apply fading memory and scale the covariance matrix
            self.sig_ukf[:, :, ti] = self.sigma_scalar[0,ti]*self.sig_ukf[:, :, ti]

            """ Update Step """
            # perform sampling of unscented points using predicted mean and variance
            X = unscentedTrans(self.lam, self.x_ukf[:, ti], self.sig_ukf[:, :, ti])

            # obtain the predicted sensor signal using the predicted mean estimate
            if self.model is not None:
                shape_matrix = np.repeat(np.expand_dims(self.shape_numpy, axis=0), X.T.shape[0], axis=0)
                shape_matrix = torch.from_numpy(shape_matrix).type(torch.FloatTensor).to(self.device)
                X_tensor = np.expand_dims(X.T, axis=1)
                X_tensor = torch.from_numpy(X_tensor).type(torch.FloatTensor).to(self.device)
                pred = self.model(X_tensor, shape_matrix).cpu().detach().numpy()
                pred = pred * (self.max_s - self.min_s) + self.min_s
                pred /= self.sensor_atten
                gt_mu_x = pred[:, 0].T
                gt_mu_y = pred[:, 1].T
            else:    
                gt_mu_x = self.mx.predict(X.T)[0].T
                gt_mu_y = self.my.predict(X.T)[0].T
            y_pred = np.vstack((gt_mu_x, gt_mu_y))

            # sigma points for state and measurement
            # perform step 8 to get weighted sum of predicted unscented points of sensor signal
            self.y_est[:, ti] = wmean(self.weight, y_pred)
            # perform step 9 to get covariance matrix within sensor signal
            self.cov_yy[:, :, ti] = wcov(self.weight, y_pred, y_pred) + self.R
            # perform step 10 to get covariance matrix between contact position and sensor signal
            self.cov_xy[:, :, ti] = wcov(self.weight, X, y_pred)
            # calculate the Kalman Gain
            K = self.cov_xy[:, :, ti].dot(np.linalg.inv(self.cov_yy[:, :, ti]))
            # update mean and variance with Kalman Gain
            self.x_ukf[:, ti] = self.x_ukf[:, ti] + K.dot(self.Y_t[ti,:].T - self.y_est[:, ti])
            self.sig_ukf[:, :, ti] = self.sig_ukf[:, :, ti] - K.dot(self.cov_yy[:, :, ti]).dot(K.T)
            self.sig_sqrtm[:, :, ti] = sqrtm(self.sig_ukf[:, :, ti])

        other_data = {
            'X_t' : self.X_t,
            'Y_t' : self.Y_t,
            'Xnew': Xnew_save,
            'selection_arr' : selection_arr,
            'sig_sqrtm' : self.sig_sqrtm,
            'mu0': self.mu0,
        }

        return (self.x_ukf, self.sig_ukf, other_data)

if __name__ == "__main__":

    data_tbl = pd.read_csv('../data/all_datasets/test/tip_test_04_26_23/tip_test_point_sensor1_April_26.csv')

    proprio_state = np.vstack((data_tbl.px.to_numpy(),
                            data_tbl.py.to_numpy(),
                            data_tbl.pz.to_numpy())).T


    sensor_signal = np.vstack((data_tbl.sx.to_numpy(),
                            data_tbl.sy.to_numpy(),
                            data_tbl.sz.to_numpy())).T

    sensor_atten = 50
    sensor_signal = sensor_signal[:,:2]/sensor_atten

    contact_loc = ContactPointTracker()

    contact_loc.initialize(proprio_state, sensor_signal)
    x_ukf, sig_ukf = contact_loc.run_tracker()

    ev_array = np.zeros(x_ukf.shape[1])
    for i in range(x_ukf.shape[1]):
        ev = np.linalg.norm(np.linalg.eig(sig_ukf[:,:,i])[0][0:2])
        ev_array[i] = ev

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')

    x_est = x_ukf.T
    x_err = contact_loc.X_t-x_est
    # x_err = x_err[np.where(ev_array < .003)[0],:]

    data_range = np.arange(180)
    # s1 = ax.scatter(x_err[:,0], -x_err[:,1], -x_err[:,2])
    s1 = ax.scatter(x_est[:,0], -x_est[:,1], -x_est[:,2])
    s2 = ax.scatter(contact_loc.X_t[:,0], -contact_loc.X_t[:,1], -contact_loc.X_t[:,2])
    ax.view_init(90, -90)

    ax.set_xlabel('x axis (mm)')
    ax.set_ylabel('y axis (mm)')
    ax.set_zlabel('z axis (mm)')
    ax.set_xlim([-40,0])
    ax.set_ylim([0,20])
    ax.set_zlim([-5,5])
    plt.show()

    # generate 3D surface map