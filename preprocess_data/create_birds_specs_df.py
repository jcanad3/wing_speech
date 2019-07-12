import pandas as pd

with open('birds.csv', 'r') as f:
	species = f.read().splitlines()

species = species[1:]
ids = list(range(1, len(species) + 1))

data = list(zip(ids, species))

df = pd.DataFrame(data, columns=['species_id', 'species_english'])
print(df)
