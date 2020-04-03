import matplotlib.pyplot as plt
import os
from util import load_batch
from sklearn.decomposition import PCA

embed_files = os.listdir("faces")
reference_embeds = load_batch(embed_files,folder_prefix="faces")
labels = [embed_file.split('.')[0] for embed_file in embed_files]

reference_embeds = [reference_embed.flatten() for reference_embed in reference_embeds]

pca = PCA(n_components=2)
transformed = pca.fit_transform(reference_embeds)

plt.scatter(transformed[:,0],transformed[:,1])
for i, coord in enumerate(transformed):
  text = labels[i]
  plt.annotate(text,coord)
  
plt.show()




