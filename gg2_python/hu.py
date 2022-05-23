import numpy as np
from sklearn.linear_model import HuberRegressor
from attenuate import *
from ct_calibrate import *
from ct_phantom import *


def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use water to calibrate
	n = reconstruction.shape[1]
	water = material.coeff('Water')
	depth = 2 * n * scale
	
	I_water = attenuate(p, water, depth)
	reconstruction_water = -np.log(reconstruction/np.sum(I_water))
	# put this through the same calibration process as the normal CT data
	reconstruction_water_calibrated = ct_calibrate(p, material, reconstruction_water, scale)
	# use result to convert to hounsfield units
	HU = (reconstruction - reconstruction_water_calibrated)/reconstruction_water_calibrated * 1000
	# limit minimum to -1024, which is normal for CT data.
	c = 0
	w = 200
	reconstruction = ((HU - c)/w)*128 + 128

	return reconstruction