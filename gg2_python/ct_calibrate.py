import numpy as np
import scipy
from scipy import interpolate
from attenuate import attenuate

def ct_calibrate(photons, material, sinogram, scale):

	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""

	# Get dimensions and work out detection for just air of twice the side
	# length (has to be the same as in ct_scan.py)
	n = sinogram.shape[1]
	air = material.coeff('Air')
	depth = 2 * n * scale
	# Attenuations through pure air give the same result from every direction,
	# due to fixed distance (depth) between transmitter and receiver.
	I_0 = attenuate(photons, air, depth)

	# Beam hardening correction
	water = material.coeff('Water')
	T_w = np.linspace(0,50)
	P_w = [attenuate(photons, water, t_w) for t_w in T_w]
	print(P_w)
	params = np.polyfit(T_w, P_w, 3)
	C = P_w[0]/T_w[1]
	sinogram = C*np.polyval(params,sinogram)

	# perform calibration
	sinogram = -np.log(sinogram/np.sum(I_0))

	return sinogram