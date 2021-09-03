import sys
sys.path.insert(0, './')
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# code to get embeddings from file
def load_embeddings():
    # TODO

data = load_embeddings()

pca = PCA(n_components = 50)

data_reduced = pca.fit_transform(data)

tsne = TSNE(n_components = 2, verbose=1, perplexity=30, n_iter=300)
data_tsne = tsne.fit_transform(data_reduced)

data_2d_one = data_tsne[:,0]
data_2d_two = data_tsne[:,1]


plt.scatter(data_2d_one, data_2d_two)
plt.savefig('features_tsne_vis.png')
