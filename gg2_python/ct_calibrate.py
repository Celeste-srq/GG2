import numpy as np
import scipy
from scipy import interpolate
from attenuate import attenuate
from ct_detect import ct_detect

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

	# perform calibration
	sinogram = -np.log(sinogram/np.sum(I_0))

	print(sinogram[128, :])
	print("")
	# Beam hardening correction
	water = material.coeff('Water')

	T_w = np.linspace(1,n,n)*scale
	P_w = -np.log(ct_detect(photons, water, T_w)/np.sum(I_0))
	sinogram_t = np.interp(sinogram,P_w,T_w)
	print(P_w)
	C = sinogram[0,0]/sinogram_t[0,0]
	sinogram = C*sinogram_t
	print("")
	print(sinogram[128, :])

	return sinogram