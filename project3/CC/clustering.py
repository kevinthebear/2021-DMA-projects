from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.cluster import KMeans
import os

categories = ['opinion', 'business', 'world', 'us', 'arts', 'sports', 'books', 'movies']

data = load_files(container_path='/Users/minseop/Desktop/2021-2/데이터관리와 분석/프로젝트 #3/DMA_project3/DMA_project3/CC/text_all', categories=categories, shuffle=True,
                    encoding='utf-8', decode_error='replace')

# Data preprocessing and clustering
data_trans = TfidfTransformer().fit_transform(CountVectorizer(stop_words='english', min_df=4, max_df=.3, 
                                                              max_features=1000).fit_transform(data.data))
clst = KMeans(n_clusters=8, max_iter=10000, init='random', 
              n_init=30, algorithm='full', random_state=0, tol=1e-4)
clst.fit(data_trans)

print(metrics.v_measure_score(data.target, clst.labels_))


from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

clusters = clst.labels_.tolist()
labels = data.target
colors = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: 'darkgoldenrod'}

pca = PCA(n_components=2).fit_transform(data_trans.toarray())
xs, ys = pca[:, 0], pca[:, 1]
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))
#df = pd.DataFrame(dict(x=xs, y=ys, label=labels))
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17,9))
ax.margins(0.05)

for idx, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, color=colors[idx], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    ax.tick_params(
        axis='y',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')

plt.show()
