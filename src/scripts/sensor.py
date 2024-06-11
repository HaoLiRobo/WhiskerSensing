import numpy as np
import GPy
import pandas as pd

class Sensor:
    """ 
    Sensor class used to load sensor parameters
    and hold sensor model
    """

    def __init__(self, sensor_num, location=None, kernel=None):
        """
        sensor_num: int
        location: tuple (x,y,z)
        """
        self.data = np.load(f'../data/sensor_models/sensor{sensor_num}/sensor{sensor_num}_models.npz', allow_pickle=True)
        self.P = self.data['P']
        self.S = self.data['S']
        self.sensor_shape=self.data['shape']

        self.pos_ref = pd.read_csv(f'../data/sensor_models/sensor{sensor_num}/sensor_pos_ref.csv').to_numpy()

        if kernel is None:
            kernel = GPy.kern.ThinPlate(3, variance=1.0, R=100)
        self.mx = GPy.models.GPRegression(self.P,self.S[:,0:1],kernel)
        self.my = GPy.models.GPRegression(self.P,self.S[:,1:2],kernel)


if __name__ == "__main__":
     s=Sensor(sensor_num=1)
     print(s.sensor_shape)