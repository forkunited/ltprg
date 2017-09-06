import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_embedding(embedding, rgbs, name, save_path):
	assert embedding.shape[1] == 2 or embedding.shape[1] == 3
	fig = plt.figure(facecolor='white')
	if embedding.shape[1] == 3:
		ax = Axes3D(fig)
		ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
				   c=rgbs, s=500, depthshade=False)
	else:
		plt.scatter(embedding[:, 0], embedding[:, 1], 
					c=rgbs, s=500)
	plt.axis('tight')
	plt.title(name)
	plt.show()
	# plt.savefig(save_path + name + '.png')

def apply_tsne(X, name, rgbs, save_path):
	# X is a num_stims x n np array, where n is the dimensionality
	# of the high-d embedding for each stim
	# applies TSNE to get 2-dim embedding
	tsne_embedding = TSNE(n_components=3, init='pca', 
						  random_state=0).fit_transform(X)
	plot_embedding(tsne_embedding, rgbs, 'tSNE: ' + name.upper(),
				   save_path)