import numpy as np
import glob, os

for spec_path in glob.glob('../train_bird_spectrograms/*'):
	base_name = os.path.basename(spec_path)
	train_spec = np.load(spec_path)
	print('Train Spec', train_spec.shape)
	test_spec = np.load('../test_bird_spectrograms/' + base_name)
	print('Test Spec', test_spec.shape)

	new_spec = np.vstack((train_spec, test_spec))
	print('New Spec', new_spec.shape)
	np.save(spec_path, new_spec)
	print('')
