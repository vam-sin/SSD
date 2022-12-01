# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
sns.set_theme()
sns.set(font="Verdana")

# import data embeddings
embeds_ID_train = np.load('embeds/features_train.npz', allow_pickle = True)['arr_0']
embeds_ID_test = np.load('embeds/features_test.npz', allow_pickle = True)['arr_0']
embeds_OOD_k = np.load('embeds/features_ood_k.npz', allow_pickle = True)['arr_0']
embeds_OOD_not_k = np.load('embeds/features_ood_not_k.npz', allow_pickle = True)['arr_0']

total_embeds = np.concatenate([embeds_ID_train, embeds_ID_test, embeds_OOD_k, embeds_OOD_not_k], axis=0)

print(embeds_ID_train.shape, embeds_ID_test.shape, embeds_OOD_k.shape, embeds_OOD_not_k.shape, total_embeds.shape)

colors_annots = []

for i in range(len(embeds_ID_train)):
    colors_annots.append('#eb2f06')

for i in range(len(embeds_ID_test)):
    colors_annots.append('#f6b93b')

for i in range(len(embeds_OOD_k)):
    colors_annots.append('#1e3799')

for i in range(len(embeds_OOD_not_k)):
    colors_annots.append('#78e08f')

print(len(colors_annots))

# plot a UMAP of the embeds
reducer = umap.UMAP()
low_dim_embeds = reducer.fit_transform(total_embeds)
print(low_dim_embeds.shape)
# low_dim_embeds = embeds_vae

# plot
plt.scatter(
    low_dim_embeds[:, 0],
    low_dim_embeds[:, 1],
    c=colors_annots)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP of SSD: BG and Microtubule', fontsize=20)
# plt.show()
plt.savefig('img/umap.png')
