#!/usr/bin/env python3

from torch.utils.data import DataLoader
from hparam import hparam as hp
from data_load import BirdSongData
from speech_embedder_net import SpeechEmbedder
from batch_all_triplet_loss import batch_all_triplet_loss, batch_all_gaussian_triplet_loss
import numpy as np
import pandas as pd
import os, random, time, torch, gc, glob, math

# GAT = Gaussian Triplet
def get_gaussian_triplet_batch(df_rec_and_species, spec_paths):
	# spectrograms are individual recordings in this case
	# needs a spectrogram, a spectrogram_id (recording_label), and a species id (species_label)
	
	spec_arrs = np.asarray(())
	recording_labels = np.array([])
	species_labels = np.array([])

	recording_label = 0
	for spec_path in spec_paths:
		spec = np.load(spec_path)
		rec_id = os.path.basename(spec_path).replace('.npy', '')
		rec_id = int(rec_id)

		if spec.shape[0] >= 2:
			# get a max of twenty samples from each spectrogram
			if spec.shape[0] > 20:
				max_samps_per_spk = 20
				num_samps = max_samps_per_spk
				idxs = np.random.randint(0, spec.shape[0], max_samps_per_spk)
				spec = spec[idxs, :]
			else:
				num_samps = spec.shape[0]	
	
			if spec_arrs.size == 0:
				spec_arrs = spec
			else:
				spec_arrs = np.vstack((spec_arrs, spec))
	
	
			# find species label using df
			species_id = df_rec_and_species.loc[df_rec_and_species['recording_id'] == rec_id, 'species_id'].values[0]
			species_id = int(species_id)
			species_repeats = np.repeat(species_id, num_samps)
			species_labels = np.append(species_labels, species_repeats)
	
			# add recording labels for embs
			spec_labels = np.repeat(recording_label, num_samps)
			recording_labels = np.append(recording_labels, spec_labels)
	
			recording_label += 1

	recording_labels = recording_labels.astype(int)
	species_labels = species_labels.astype(int)

	print('Spec Arrs shape', spec_arrs.shape)
	print('Recording labels', recording_labels.shape)
	print('Species Labels', species_labels.shape)

	return spec_arrs, recording_labels, species_labels

def train(model_path):
	device = torch.device(hp.device)

	train_dataset = BirdSongData()

	# pytorch data loader with batch size of 4 (speakers)
	train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
	
	# architecture of LSTM net
	embedder_net = SpeechEmbedder().to(device)

	# restore from previous training section
	#if hp.train.restore:
	#embedder_net.load_state_dict(torch.load('speech_id_checkpoint/ckpt_epoch_105_batch_id_243.pth'))
	
	#Both net and loss have trainable parameters
	optimizer = torch.optim.Adam([
					{'params': embedder_net.parameters()},
				], lr=hp.train.lr)
	
	os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
	
	# set train mode for embedder net
	embedder_net.train()
	iteration = 0

	df_rec_and_species = pd.read_csv('../preprocess_data/species_and_record_url.csv')	
	bird_spec_paths = glob.glob('../single_bird_spectrograms/*')

	# train for number of epochs
	for e in range(0, hp.train.epochs):
		random.shuffle(bird_spec_paths)
		batch_id = 0
		for spec_idx in range(0, len(bird_spec_paths), 10):
			specs_arr, recording_labels, species_labels = get_gaussian_triplet_batch(
				df_rec_and_species,
				bird_spec_paths[spec_idx:spec_idx+10]
			)
			
			# convert data and labels to tensors
			mel_db_batch = torch.from_numpy(specs_arr)
			recording_labels = torch.from_numpy(recording_labels)
			species_labels = torch.from_numpy(species_labels)			

			# move mel batch to device (cpu or gpu)	
			mel_db_batch = mel_db_batch.to(device)

			# num frames
			#num_frames = np.random.randint(24, 160, size=1)
			#num_frames = num_frames[0]
			#print('Num frames for batch', num_frames)
			num_frames = 20
			
			# offset
			#offset = np.random.randint(0, 160 - num_frames, size=1)
			#offset = offset[0]
			#print('Offset', offset)	

			# convert mel_db_batch into 3-D array (batch, timestepts, logmels)
			#mel_db_batch = mel_db_batch[:, offset:offset+num_frames, :]
			mel_db_batch = mel_db_batch[:, :, :]

			#gradient accumulates
			optimizer.zero_grad()
			
			# fit embedding net to current mel batch
			embeddings = embedder_net(mel_db_batch)
	
			# triplet loss and fraction of positive	
			loss, fraction_of_positive = batch_all_gaussian_triplet_loss(embeddings, recording_labels, species_labels) 
			loss.backward()

			# optimization step
			optimizer.step()
			
			fraction_of_positive = fraction_of_positive.detach()

			iteration += 1

			# print the training info after log_interval iterations
			if (batch_id + 1) % hp.train.log_interval == 0:
				mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tBatch-Loss:{5:.4f}\tFraction-Positive:{6:.4f}\t\n".format(time.ctime(), e+1,
						batch_id + 1, math.ceil(len(bird_spec_paths)/10) , iteration,loss, fraction_of_positive)
				print(mesg)
				if hp.train.log_file is not None:
					with open(hp.train.log_file,'a') as f:
						f.write(mesg)

			batch_id += 1

		# save the model parameters at the checkpoint
		if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
			embedder_net.eval().cpu()
			ckpt_model_filename = "gt_ckpt_epoch_" + str(e+1) + "_batch_id_" + str(batch_id+1) + ".pth"
			ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
			torch.save(embedder_net.state_dict(), ckpt_model_path)
			embedder_net.to(device).train()

	#save model after all epochs
	embedder_net.eval().cpu()
	save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
	save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
	torch.save(embedder_net.state_dict(), save_model_path)
	
	print("\nDone, trained model saved at", save_model_path)
		
if __name__=="__main__":
	if hp.training:
		train(hp.model.model_path)
	#else:
	#	test(hp.model.model_path)

