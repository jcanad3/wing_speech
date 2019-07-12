#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
from hparam import hparam as hp
from vad import vad
import numpy as np
import glob, os, librosa

def gen_spectrogram():
	for bird_type in glob.glob('../birds_mp3_wavs/*'):
		bird_name = os.path.basename(bird_type)
		print('Bird:', bird_name)
		if len(os.listdir(bird_type)) > 0:
			spec_samples = []
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
			
					for idx in range(0, S.shape[0], 160):
						samp = S[idx:idx+160, :]
						if samp.shape == (160, 40):
							spec_samples.append(samp)
				except:
					print('Error creating spectrogram')
				
				#  remove voice only file		
				os.remove(voice_only_path)
			
			spec_samples = np.array(spec_samples)
			if spec_samples.shape[0] > 2:
				train_spec_samples = spec_samples[:int(0.8*spec_samples.shape[0]), :, :]
				test_spec_samples = spec_samples[int(0.8*spec_samples.shape[0]):, :, :]

				print('\t\t - Train Log Melspectrogram', train_spec_samples.shape)
				print('\t\t - Test Log Melspectrogram', test_spec_samples.shape)

				if train_spec_samples.shape[0] > 0:
					np.save('../train_bird_spectrograms/' + bird_name + '.npy', train_spec_samples)
				if test_spec_samples.shape[0] > 0:
					np.save('../test_bird_spectrograms/' + bird_name + '.npy', test_spec_samples)

			print('')	

if __name__ == "__main__":
	gen_spectrogram()
