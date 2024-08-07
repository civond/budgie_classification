import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

json_dir = "json/"
files = os.listdir(json_dir)
print(len(files))

data = []
label = []
i = 0
for file in files:
    
    temp_path = os.path.join(json_dir, file)
    df = pd.read_json(temp_path)
    temp_activation = np.array(df['activations'].tolist(), dtype=np.float32)
    temp_label = df['label'][0]

    
    flattened_data = [array.flatten() for array in temp_activation]
    #print(flattened_data)
    X = np.array(flattened_data)
    X = X.T
    data.append(X)
    label.append(temp_label)


combined_array = np.vstack(data)
label = np.array(label)
print(label.shape)
print(combined_array.shape)

n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(combined_array)

tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': label})
print(tsne_result_df)
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', 
                y='tsne_2', 
                hue='label', 
                data=tsne_result_df, 
                ax=ax,
                s=15)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()

    #print(data.shape)
"""df = pd.read_json("json/0.json")
df2 = pd.read_json("json/1.json")
df3 = pd.read_json("json/2.json")
df4 = pd.read_json("json/3.json")

temp = np.array(df['activations'].tolist(), dtype=np.float32)
temp2 = np.array(df2['activations'].tolist(), dtype=np.float32)
temp3 = np.array(df3['activations'].tolist(), dtype=np.float32)
temp4 = np.array(df4['activations'].tolist(), dtype=np.float32)

data = []
data.append(temp)
data.append(temp2)
print(data)

flattened_data = [array.flatten() for array in data]
X = np.array(flattened_data)
print(X.shape)
print(X)

tsne = TSNE(n_components=2, perplexity=3)
X_tsne = tsne.fit_transform(X)

# Plot the result
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()"""