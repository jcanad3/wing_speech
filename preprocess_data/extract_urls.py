import numpy as np
import pandas as pd

df = pd.read_csv('Occurrence.txt')

orig_s_and_urls = df.loc[:, ['vernacularName', 'associatedMedia']].values.tolist()
species_keys_df = pd.read_csv('species_keys.csv')

rec_ids = []
mp3_urls = []
species_ids = []
count = 0
for name, url in orig_s_and_urls:
	mp3 = url.split('|')[0]
	
	s_id = species_keys_df.loc[species_keys_df['common_name'] == name, 'species_id'].values
	s_id = s_id[0]
	
	rec_ids.append(count)
	mp3_urls.append(mp3)
	species_ids.append(s_id)

	print('MP3 URL: {0} \t s_id: {1} \t rec_ids: {2}'.format(mp3, s_id, count))
	count += 1


rec_ids = np.array(rec_ids)
mp3_urls = np.array(mp3_urls)
species_ids = np.array(species_ids)

data = np.column_stack((rec_ids, mp3_urls, species_ids))
new_df = pd.DataFrame(data, columns=['recording_id', 'recording_url', 'species_id'])
print(new_df)
new_df.to_csv('species_and_record_url.csv')
#mp3 = [url.split('|')[0] for url in urls]
#mp3 = np.array(mp3)
#rec_ids = list(range(0, len(urls)))


#data = np.column_stack((rec_ids, species, mp3))

#species_rec = pd.DataFrame(data, columns=['recording_id', 'recording_url', 'species_id'])
#print(species_rec)

