
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

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
	# explain what this test is for
	# compare the initial image and output from scan_and_reconstruct

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 3)
	#s = source.photon('100kVp, 3mm Al')
	#use fake source with only one frequency instead of real source
	s = fake_source(source.mev, 0.07, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)*2

	# save some meaningful results
	#save_draw(p, 'results', 'test_1_phantom')
	save_draw(y, 'results', 'test_1_image', caxis=[0,max(map(max, y))])

	# how to check whether these results are actually correct?
	# get original mu(x,y) and compare with reconstructed mu(x,y)
	for i in range(p.shape[0]):
		for j in range(p.shape[1]):
			# multiply by 0.7
			p[i,j] = material.coeff(material.name[int(p[i,j])])[np.where(material.mev == 0.07*0.7)]
	
	save_draw(p, 'results', 'test_1_phantom_material')
	save_draw(np.abs(y-p), 'results', 'test_1_difference')

	# metrics for good/bad?

def test_2():
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_plot(y[128,:], 'results', 'test_2_plot')

	# how to check whether these results are actually correct?

def test_3():
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# save some meaningful results
	f = open('results/test_3_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.close()

def test_4():
	# explain what this test is for
	#plot the attenuation y values of the phantom and the reconstruction on the same graph, to observe v
	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 3)
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)
	for i in range(p.shape[0]):
		for j in range(p.shape[1]):
			# multiply by 0.7
			p[i,j] = material.coeff(material.name[int(p[i,j])])[np.where(material.mev == 0.07*0.7)]

	y_phantom = p[128, :]
	y_resonctruct = y[128, :]
	# save some meaningful results
	save_comparison(y_phantom, y_resonctruct, 'results', 'test_4_attenuation_comparison', label1='phantom value', label2='reconstructed value')
	
	# draw(y)
	# print(y.shape)
	# print(y)
	


def test_5():
	# explain what this test is for
	# test if the scan and reconstructed value matches the theoretical attenuated value for the material
	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	coeff = material.coeff('Soft Tissue')[np.where(material.mev ==0.07)]
	



	# save some meaningful results
	save_draw(p, 'results', 'test_5_phantom')
	f = open('results/test_5_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.write('Expected value is ' + str(coeff))
	f.close()


# Run the various tests
# print('Test 1')
# test_1()
# print('Test 2')
# test_2()
# print('Test 3')
# test_3()
print('Test 4')
test_4()
# print('Test 5')
# test_5()
