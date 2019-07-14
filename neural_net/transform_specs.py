import numpy as np
import glob

for spec_path in glob.glob('../test_bird_spectrograms/*'):
	spec = np.load(spec_path)

	print('Old Spec', spec.shape)

	spec = spec.reshape(spec.shape[0]*spec.shape[1], spec.shape[2])
	
	new_spec = []
	for i in range(0, spec.shape[0], 20):
		spec_samps = spec[i:i+20, :]
		new_spec.append(spec_samps)

	new_spec = np.array(new_spec)
	print('New Spec', new_spec.shape)
	np.save(spec_path, new_spec)
