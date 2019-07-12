import numpy as np
import pandas as pd

df = pd.read_csv('Occurrence.txt')

species = df['vernacularName'].values

species = np.unique(species)

species_ids = list(range(0, species.shape[0]))
species_ids = np.array(species_ids)

data = np.column_stack((species_ids, species))

new_df = pd.DataFrame(data, columns=['species_id', 'common_name'])
print(new_df)

new_df.to_csv('species_keys.csv')
