#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import numpy as np
from hparam import hparam as hp
from vad import vad

# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))										

def save_spectrogram_tisv():
	""" Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
		Each partial utterance is splitted by voice detection using DB
		and the first and the last 180 frames from each partial utterance are saved. 
		Need : utterance data set (VTCK)
	"""
	print("start text independent utterance feature extraction")
	os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
	os.makedirs(hp.data.test_path, exist_ok=True)	# make folder to save test file
	
	for bird_type in glob.glob('../bird_species/*')
		bird_name = os.path.basename(bird_type)
		print('Bird:', bird_name)
		for recording in glob.glob(bird_type + '/*'):
			print('\t - ', recording)
			try:
				# do vad stuff
				voice_only_path = vad(recording, 2)
		
				full_song, sr = librosa.core.load(voice_only_path, sr=16000)
				S = librosa.core.stft(y=full_song, n_fft=hp.data.nfft,
									  win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
				S = np.abs(S) ** 2
				mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
				S = np.log10(np.dot(mel_basis, S) + 1e-6)		   # log mel spectrogram of utterances
				S = S.T		
		
				spec_samples = []
				for idx in range(0, S.shape[1], 160):
					samp = S[idx:idx+160, :]
					if samp.shape == (160, 40):
						spec_samples.append(samp)
				spec_samples = np.array(spec_samples)	
				print('Log Melspectrogram', spec_samples.shape)
				np.save('fma_md_specs/' + np_name, spec_samples)
		
			except:
				print('Error creating spectrogram')

		os.remove(voice_only_path)

if __name__ == "__main__":
	save_spectrogram_tisv()
