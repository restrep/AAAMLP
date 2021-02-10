import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn import manifold

data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
pixel_values, targets = data
targets = targets.astype(int)
 # pixel_values is a 2-dimensional array of shape 70000x784. 
 # There are 70000 different images, each of size 28x28 pixels. 
 # Flattening 28x28 gives 784 data points
single_image = pixel_values[1,:].reshape(28, 28)
plt.imshow(single_image, cmap='gray')

tsne = manifold.TSNE(n_components=2, random_state=42) # 2 components for visualizaiton
transformed_data = tsne.fit_transform(pixel_values[:3000, :])

tsne_df = pd.DataFrame(np.column_stack((transformed_data, targets[:3000])), columns=['x', 'y', 'targets'])

tsne_df.loc[:,'targets'] = tsne_df.targets.astype(int)

grid = sns.FacetGrid(tsne_df, hue='targets', height= 8)
grid.map(plt.scatter, 'x', 'y').add_legend()