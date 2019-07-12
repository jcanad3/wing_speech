#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils import mfccs_and_spec

class BirdSongData(Dataset):
	
	def __init__(self, shuffle=True, utter_start=0):
		
		# data path
		if hp.training:
			# training path
			self.path = hp.data.train_path
			# number of uterrances per speaker
			self.utter_num = hp.train.M
		else:
			# test specs path
			self.path = hp.data.test_path
			# number of utterances per speaker
			self.utter_num = hp.test.M
		self.file_list = os.listdir(self.path)
		self.shuffle=shuffle
		self.utter_start = utter_start
		
	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
	   
		# all spectrogram files 
		np_file_list = os.listdir(self.path)
		
		# shuffle data
		if self.shuffle:
			while True:
				selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
				utters = np.load(os.path.join(self.path, selected_file))		# load utterance spectrogram of selected speaker
				if utters.size > 0:
					break
		else:
			selected_file = np_file_list[idx]			   
		
		if self.shuffle:
			utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
			utterance = utters[utter_index]	   
		else:
			utterance = utters[self.utter_start: self.utter_start+self.utter_num] # utterances of a speaker [batch(M), n_mels, frames]

		utterance = utterance[:,:160,:]			   # TODO implement variable length batch size

		return utterance
