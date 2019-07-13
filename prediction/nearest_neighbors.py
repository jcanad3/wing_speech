from sklearn.neighbors import NearestNeighbors as NN
import numpy as np
import pandas as pd
import glob, os

df_song_species = pd.read_csv('../preprocess_data/species_and_record_url.csv')
species = pd.read_csv('../preprocess_data/species_keys.csv')

emb_ids = []
avg_embs = []
for avg_emb_path in glob.glob('../avg_bird_embeddings/*'):
	emb_id = os.path.basename(avg_emb_path).replace('.npy','')
	emb_id = int(emb_id)

	# load d-vector
	avg_emb = np.load(avg_emb_path)

	emb_ids.append(emb_id)
	avg_embs.append(avg_emb)	

avg_embs = np.array(avg_embs)

# fit nearest_neighbors
nn = NN(n_neighbors=6, metric='euclidean', n_jobs=-1)
nn.fit(avg_embs)

dists, inds = nn.kneighbors(avg_embs)
data_neighbors = []
count = 0
for neighbor_matches in inds.tolist():
	neighbor_row = []
	for n in range(0, len(neighbor_matches)):
		emb_id = emb_ids[neighbor_matches[n]]

		# get species of embedding
		species_id = df_song_species.loc[df_song_species['recording_id'] == emb_id, 'species_id'].values[0]
		species_name = species.loc[species['species_id'] == species_id, 'common_name'].values[0]
		neighbor_row.append(str(emb_id) + '.wav')
		neighbor_row.append(species_id)
		neighbor_row.append(species_name)

	# append neighbors row to main data list
	data_neighbors.append(neighbor_row)

cols = [
	'Orignal_Recording', 'Original_Species_Id', 'Original_Common_Name',
	'1st_Neighbor_Recording', '1st_Neighbor_Species_Id', '1st_Neighbor_Common_Name',
	'2nd_Neighbor_Recording', '2nd_Neighb2ndSpecies_Id', '2nd_Neighbor_Common_Name',
	'3rd_Neighbor_Recording', '3rd_Neighbor_Species_Id', '3rd_Neighbor_Common_Name',
	'4th_Neighbor_Recording', '4th_Neighbor_Species_Id', '4th_Neighbor_Common_Name',
	'5th_Neighbor_Recording', '5th_Neighbor_Species_Id', '5th_Neighbor_Common_Name']

df_neighbors = pd.DataFrame(data_neighbors, columns=cols)
print(df_neighbors)
