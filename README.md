## Wing Speech
PyTorch Triplet Loss based RNN for identifying bird species by audio recordings.

## Data
The bird song recordings from this project come xeno-canto corpus, which, as of this writing, has 472,000+ recordings and 10000+ species.
A subset of 35,183 files are used for training the RNN.

## Neural Network
The architecture and hyperparameters for the RNN come from "Speaker Diarizatio with LSTM" by Wang et al. There are 
three LSTM layers, each with 768 cells. A final linear layer contains 256 nodes, the outputs of which serve as the embedding.
For each clip, an embedding is calculated from each 200ms chunk. Once all of the embeddings for a single clip have
been calculated, the mean of this embedding serves as a vector representation of the full clip.

## Prediction Methodologies
When predicting the species for a new recording, the feature extraction process is identical the one mentioned above, i.e.
the average embedding is calculated for each file. Three methods have been implemented to predict the class: Nearest Embedding
Classification, Nearest Centroid Classification, and K-Nearest Neighbors Classification. Prior to prediction, the embedding 
for each file in the training set is calculated. This process is quite time consuming, and can take several hours to complete. Upon
cessation, each file in the test set is passed to the above mentioned classifiers. For the Nearest Embedding Classification, the top-1
and the top-5 accuracy is calculated, while the other methods only predict the top-1 accuracy, and return a classification report
containg the precision, recall, and F1 scores for each class. 

## Conclusion

