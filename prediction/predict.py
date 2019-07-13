from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

song_species = pd.read_csv('../preprocess_data/species_and_record_url.csv')
data = pd.read_csv('labeled_bird_embs.csv')
X = data.loc[:, 'x_0':'x_255'].values
y = data['label'].values

nn = NN(n_neighbors=6, n_jobs=-1)
nn.fit(X)
dists, inds = nn.kneighbors(X)
print('INDS', inds)
print('Dists', dists)
