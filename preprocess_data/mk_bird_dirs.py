import numpy as np
import pandas as pd
import os

df = pd.read_csv('species_keys.csv')
species_keys = df['species_id'].values.tolist()

for bird_id in species_keys:
	os.makedirs('../birds_mp3_wavs/' + str(bird_id), exist_ok=True)
