import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)

	# apply filter to all angles
	print('Ramp filtering')

	# sample frequency list for fft
	w_list = np.fft.fftfreq(m, d=scale)*2*np.pi
	wmax = max(w_list)
	# define filter
	ram_lak = []
	for w in w_list:
		ram_lak.append((np.abs(w)/(2*np.pi)))
	
	#ram_lak = np.array([np.abs(w)/(2*np.pi) for w in w_list])
	# replace zero at k=0 with (1/6) of the value at k=1, where k is the frequency index
	ram_lak[0] = (1/6)*ram_lak[1]
	ram_lak = np.array(ram_lak)

	# 1D Fourier transform on sinogram in r direction
	fft_sinogram = np.empty((angles,m), dtype=complex)
	for i in range(angles):
		fft_sinogram[i] = np.fft.fft(sinogram[i], n=m)

	# apply filter
	fft_sinogram *= ram_lak

	# inverse Fourier transform on sinogram in r direction and take first n
	ifft_sinogram = np.zeros((angles,n))
	for i in range(angles):
		ifft_sinogram[i] = np.fft.ifft(fft_sinogram[i], n=m).real[:n]
	
	return ifft_sinogram