import numpy as np
import glob, os

for spec_path in glob.glob('../train_2_birds_spectrograms/*'):
	base_name = os.path.basename(spec_path)
	train_spec = np.load(spec_path)
	if train_spec.shape[0] > 0:
		print('Train Spec', train_spec.shape)
		if base_name in os.listdir('../train_bird_spectrograms'):
			orig_spec = np.load('../train_bird_spectrograms/' + base_name)
			print('Orig Spec', orig_spec.shape)
	
			new_spec = np.vstack((orig_spec, train_spec))
			print('New Spec', new_spec.shape)
			np.save('../train_bird_spectrograms/' + base_name, new_spec)
			print('')
		else:
			print('[+] New Species [+]')
			print('Train Spec', train_spec.shape)
			np.save('../train_bird_spectrograms/' + base_name, train_spec)
			print('')
	else:
		print('Spectrogram has 0 samples.')
		print('')
# loop over new spectrograms 
# if spectrogam has partner in train_bird_spectrograms append it to that partner
# save it to train_bird_spectrom
