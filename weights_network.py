import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils

fig, (layer1, layer2) = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.01)
_, _, _, right = utils.load()


# layer0.set_title('Before')
# sns.heatmap(right[0:100, 0:100], xticklabels=False, yticklabels=False, ax=layer0, cmap='viridis', cbar=False)
# fig.colorbar(layer0.collections[0], ax=layer0, location="left", use_gridspec=False, pad=0.05)

layer1.set_title('weights of layer 1')
layer1_weights = np.loadtxt('output/l1_weights.txt', delimiter=",")
layer1_weights = np.multiply(right, layer1_weights)
sns.heatmap(layer1_weights[0:100, 0:100], xticklabels=False, yticklabels=False, ax=layer1, cmap='viridis', cbar=False)
fig.colorbar(layer1.collections[0], ax=layer1, location="left", use_gridspec=False, pad=0.05)

layer2.set_title('weights of layer 2')
layer2_weights = np.loadtxt('output/l2_weights.txt', delimiter=",")
sns.heatmap(layer2_weights[0:100, ], xticklabels=False, yticklabels=False, ax=layer2, cmap='viridis', cbar=False)
fig.colorbar(layer2.collections[0], ax=layer2, location="right", use_gridspec=False, pad=0.05)

plt.show()
