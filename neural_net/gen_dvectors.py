#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
from hparam import hparam as hp
from vad import vad
import numpy as np
from speech_embedder_net import SpeechEmbedder
import glob, os, librosa, torch

def gen_spectrogram(d_vector_model, audio_dir, avg_emb_dir):
	for bird_type in glob.glob('../' + audio_dir + '/*'):
		bird_name = os.path.basename(bird_type)
		print('Bird:', bird_name)
		if len(os.listdir(bird_type)) > 0:
			for recording in glob.glob(bird_type + '/*'):
				rec_name = os.path.basename(recording)
				if rec_name.replace('wav', 'npy') not in os.listdir('../' + avg_emb_dir):
					print('\t - ', recording)
					try:
						# do vad stuff
						voice_only_path = vad(recording, 3)
				
						full_song, sr = librosa.core.load(voice_only_path, sr=16000)
						S = librosa.core.stft(y=full_song, n_fft=hp.data.nfft,
											  win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
						S = np.abs(S) ** 2
						mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
						S = np.log10(np.dot(mel_basis, S) + 1e-6)		   # log mel spectrogram of utterances
						S = S.T		
				
						spec_samples = []
						for idx in range(0, S.shape[0], 20):
							samp = S[idx:idx+20, :]
							if samp.shape == (20, 40):
								spec_samples.append(samp)
					except:
						print('Error creating spectrogram')
					
					#  remove voice only file		
					os.remove(voice_only_path)
				
					spec_samples = np.array(spec_samples)
					if spec_samples.shape[0] > 0:
						if spec_samples.shape[0] > 500:
							spec_samples = spec_samples[:500, :]
						print('\t\t - Log Melspectrogram', spec_samples.shape)
						spec_samps = torch.from_numpy(spec_samples)
						embs = d_vector_model(spec_samps).detach().numpy()
						avg_embs = np.mean(embs, axis=0)
						np.save('../' + avg_emb_dir + '/' + rec_name.replace('wav', 'npy'), avg_embs)
				else:
					print('File already exists')

				print('')	

if __name__ == "__main__":
	d_vector_model = SpeechEmbedder()
	d_vector_model.load_state_dict(torch.load('speech_id_checkpoint/ckpt_epoch_230_batch_id_529.pth'))
	d_vector_model.eval()
	audio_dir = 'train_2_birds_wavs'
	avg_emb_dir = 'avg_test_bird_embs'

	gen_spectrogram(d_vector_model, audio_dir, avg_emb_dir)
