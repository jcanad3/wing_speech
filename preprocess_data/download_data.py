from multiprocessing import Pool
import numpy as np
import pandas as pd
import requests, os

def fetch_recording(rec_id, url, species_id):
	os.makedirs('../new_test_files_mp3/' + str(species_id), exist_ok=True)
	if str(rec_id) + '.wav' not in os.listdir('../birds_wavs/' + str(species_id)) \
		and str(rec_id) + '.wav' not in os.listdir('../train_2_birds_wavs/' + str(species_id)):
		req = requests.get(url)
		write_path = '../new_test_files_mp3/' + str(species_id) + '/' + str(rec_id) + '.mp3'
		print(write_path)
		with open(write_path, 'wb') as f:
			f.write(req.content)
		print('Downloaded Bird Species: {0}, Url: {1}'.format(species_id, url))
	else:
		print('Already downloaded file.')
	print('')

df = pd.read_csv('species_and_record_url.csv')
data = df.loc[:, ['recording_id', 'recording_url', 'species_id']].values
rec_ids = data[:, 0].tolist()
recording_url = data[:, 1].tolist()
species_ids = data[:, 2].tolist()

pool = Pool(8)
results = pool.starmap(fetch_recording, zip(rec_ids, recording_url, species_ids)) 	
