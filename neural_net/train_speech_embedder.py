#!/usr/bin/env python3

from torch.utils.data import DataLoader
from hparam import hparam as hp
from data_load import BirdSongData
from speech_embedder_net import SpeechEmbedder
from batch_all_triplet_loss import batch_all_triplet_loss
import numpy as np
import os, random, time, torch, gc, glob

def get_batch(spec_paths):
	spec_arrs = np.asarray(())
	labels = np.array([])

	label = 0
	for spec_path in spec_paths:
		spec = np.load(spec_path)

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

		spec_labels = np.repeat(label, num_samps)
		labels = np.append(labels, spec_labels)

		label += 1

	spec_arrs = np.array(spec_arrs)
	labels = labels.astype(int)
	
	return spec_arrs, labels

def train(model_path):
	device = torch.device(hp.device)

	train_dataset = BirdSongData()

	# pytorch data loader with batch size of 4 (speakers)
	train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
	
	# architecture of LSTM net
	embedder_net = SpeechEmbedder().to(device)

	# restore from previous training section
	#if hp.train.restore:
	#	embedder_net.load_state_dict(torch.load(model_path))
	
	#Both net and loss have trainable parameters
	optimizer = torch.optim.Adam([
					{'params': embedder_net.parameters()},
				], lr=hp.train.lr)
	
	os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
	
	# set train mode for embedder net
	embedder_net.train()
	iteration = 0
	
	bird_spec_paths = glob.glob('../train_bird_spectrograms/*')

	# train for number of epochs
	for e in range(hp.train.epochs):
		random.shuffle(bird_spec_paths)
		batch_id = 0
		for spec_idx in range(0, len(bird_spec_paths), 10):
			specs_arr, labels = get_batch(bird_spec_paths[spec_idx:spec_idx+10])
			mel_db_batch = torch.from_numpy(specs_arr)
			labels = torch.from_numpy(labels)
			
			# move mel batch to device (cpu or gpu)	
			mel_db_batch = mel_db_batch.to(device)

			# num frames
			#num_frames = np.random.randint(24, 160, size=1)
			#num_frames = num_frames[0]
			#print('Num frames for batch', num_frames)
			num_frames = 20
			
			# offset
			offset = np.random.randint(0, 160 - num_frames, size=1)
			offset = offset[0]
			print('Offset', offset)	

			# convert mel_db_batch into 3-D array (batch, timestepts, logmels)
			#mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
			mel_db_batch = mel_db_batch[:, offset:offset+num_frames, :]
			print('MEL DB SHAPE:', mel_db_batch.shape)
			print('Lables Shape:', labels.shape)		

			#gradient accumulates
			optimizer.zero_grad()
			
			# fit embedding net to current mel batch
			embeddings = embedder_net(mel_db_batch)
	
			# triplet loss and fraction of positive	
			loss, fraction_of_positive = batch_all_triplet_loss(embeddings, labels) 
			loss.backward()

			# optimization step
			optimizer.step()
			
			fraction_of_positive = fraction_of_positive.detach()

			iteration += 1

			# print the training info after log_interval iterations
			if (batch_id + 1) % hp.train.log_interval == 0:
				mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tBatch-Loss:{5:.4f}\tFraction-Positive:{6:.4f}\t\n".format(time.ctime(), e+1,
						batch_id + 1, round(len(bird_spec_paths)/10) , iteration,loss, fraction_of_positive)
				print(mesg)
				if hp.train.log_file is not None:
					with open(hp.train.log_file,'a') as f:
						f.write(mesg)

			batch_id += 1

		# save the model parameters at the checkpoint
		if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
			embedder_net.eval().cpu()
			ckpt_model_filename = "ckpt_epoch_" + str(e+1) + "_batch_id_" + str(batch_id+1) + ".pth"
			ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
			torch.save(embedder_net.state_dict(), ckpt_model_path)
			embedder_net.to(device).train()

	#save model after all epochs
	embedder_net.eval().cpu()
	save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
	save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
	torch.save(embedder_net.state_dict(), save_model_path)
	
	print("\nDone, trained model saved at", save_model_path)

def test(model_path):
	
	if hp.data.data_preprocessed:
		test_dataset = SpeakerDatasetTIMITPreprocessed()
	else:
		test_dataset = SpeakerDatasetTIMIT()
	test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
	
	embedder_net = SpeechEmbedder()
	embedder_net.load_state_dict(torch.load(model_path))
	embedder_net.eval()
	
	avg_EER = 0
	for e in range(hp.test.epochs):
		batch_avg_EER = 0
		for batch_id, mel_db_batch in enumerate(test_loader):
			assert hp.test.M % 2 == 0
			enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
			
			enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
			verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))
			
			perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
			unperm = list(perm)
			for i,j in enumerate(perm):
				unperm[j] = i
				
			verification_batch = verification_batch[perm]
			enrollment_embeddings = embedder_net(enrollment_batch)
			verification_embeddings = embedder_net(verification_batch)
			verification_embeddings = verification_embeddings[unperm]
			
			enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
			verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
			
			enrollment_centroids = get_centroids(enrollment_embeddings)
			
			sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
			
			# calculating EER
			diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
			
			for thres in [0.01*i+0.5 for i in range(50)]:
				sim_matrix_thresh = sim_matrix>thres
				
				FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
				/(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)
	
				FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
				/(float(hp.test.M/2))/hp.test.N)
				
				# Save threshold when FAR = FRR (=EER)
				if diff> abs(FAR-FRR):
					diff = abs(FAR-FRR)
					EER = (FAR+FRR)/2
					EER_thresh = thres
					EER_FAR = FAR
					EER_FRR = FRR
			batch_avg_EER += EER
			print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
		avg_EER += batch_avg_EER/(batch_id+1)
	avg_EER = avg_EER / hp.test.epochs
	print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
		
if __name__=="__main__":
	if hp.training:
		train(hp.model.model_path)
	else:
		test(hp.model.model_path)
