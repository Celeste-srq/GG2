# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()

def test_bh():
	# compare the initial image and output from scan_and_reconstruct

	# initial conditions
	p = ct_phantom(material.name, 256, 1)

    # real source
	s_r = source.photon('100kVp, 3mm Al')
	# fake source with only one frequency
	s_f = fake_source(source.mev, 0.07, method='ideal')
	y_r = scan_and_reconstruct(s_r, material, p, 0.01, 256)
	y_f = scan_and_reconstruct(s_f, material, p, 0.01, 256)

	save_draw(y_r, 'results', 'test_bh_image_real', caxis=[0,max(map(max, y_r))])
	save_draw(y_f, 'results', 'test_bh_image_fake', caxis=[0,max(map(max, y_f))])

	u_r = np.zeros(p.shape)
	u_f = np.zeros(p.shape)
	# get original mu(x,y) and compare with reconstructed mu(x,y)
	for i in range(p.shape[0]):
		for j in range(p.shape[1]):
			for k in range(len(s_r)):
				u_r[i,j] += material.coeff(material.name[int(p[i,j])])[np.where(material.mev == source.mev[k])] * s_r[k]
			u_r[i,j] /= np.sum(s_r)
			
			# multiply by 0.7 like in fake_source
			u_f[i,j] = material.coeff(material.name[int(p[i,j])])[np.where(material.mev == 0.07*0.7)]
		print(i)
	
	save_draw(u_r, 'results', 'test_bh_phantom_material_real')
	save_draw(u_f, 'results', 'test_bh_phantom_material_fake')

	save_draw(np.abs(y_r-u_r), 'results', 'test_bh_difference_real')
	save_comparison(u_r[128, :], y_r[128, :], 'results', 'test_bh_attenuation_comparison_real', label1='phantom value', label2='reconstructed value')
	save_draw(np.abs(y_f-u_f), 'results', 'test_bh_difference_fake')
	save_comparison(u_f[128, :], y_f[128, :], 'results', 'test_bh_attenuation_comparison_fake', label1='phantom value', label2='reconstructed value')

print('Test Beam Hardening')
test_bh()