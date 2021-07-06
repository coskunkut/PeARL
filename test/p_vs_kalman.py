import numpy as np

from persistence.generator import Multinomial
from persistence.estimator import SLWE, SLWE_P
from persistence.tools import plot_tools

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


# generate non-stationary sequence
multinomial_ins = Multinomial(25)
multinomial_ins.sudden_drift_diff(4, 2, 500, 0.3, 0.6)

# inject white noise
multinomial_ins.inject_white_noise(4, 0.3, start_index=None, stop_index=None)

f = KalmanFilter (dim_x=2, dim_z=1)
# f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
# print(f.Q)
# f.F = np.array([[1.,1.],
#                 [0.,1.]])
f.H = np.array([[1.,0.]])
f.R = 5

for token in multinomial_ins.sequence:
	f.update(token)

	f.predict()
	print(f.x)