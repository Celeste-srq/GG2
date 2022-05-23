from lib2to3.pytree import HUGE
import numpy as np
from sklearn.linear_model import HuberRegressor
from attenuate import *
from ct_calibrate import *
from ct_phantom import *
from ct_scan import *
from ct_lib import*
from ramp_filter import *
from back_project import *


def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use water to calibrate
	water_phantom = ct_phantom(material.name, 256, 1, metal='Water')
	y = ct_scan(p, material, water_phantom, 0.1, 256)
	y = ct_calibrate(p, material, y, scale)
	y = ramp_filter(y, scale, alpha=0.001)
	# Back-projection
	y = back_project(y, skip=1)
	# draw(y)
	miu_sum = 0
	num = 0
	for i in range(len(y)):
		for j in range(len(y)):
			if y[i][j] > 0:
				miu_sum += y[i][j]
				num +=1
	miu_water = miu_sum/num
	print(miu_water)
	
	# use result to convert to hounsfield units
	HU = (reconstruction - miu_water)/miu_water * 1000
	# limit minimum to -1024, which is normal for CT data.
	
	for i in range(len(HU)):
		for j in range(len(HU)):
			if HU[i][j] > 3072:
				HU[i][j] = 3072
			if HU[i][j] <-1024:
				HU[i][j] = -1024
	
	# c = 0
	# w = 200
	# reconstruction = ((HU - c)/w)*128 + 128
	reconstruction = HU

	return reconstruction