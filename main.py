import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import time

df = pd.read_csv("Your file here")

# printa informações sobre o arquivo // print file information
print(df.describe(), df.head(), df.shape, df.info())

# verificar a quantidade ideal de clusters // check the optimal number of clusters
distortions = []
Range = range(1,10)
for i in Range:
    kmeanModel = KMeans(n_clusters = i)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize = (16,8))
plt.plot(Range, distortions, "bx-")
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("optimal k")
plt.show()

# verificar o tempo de clusterização e clusteriza as amostras do documento // check the clustering time and cluster the document samples
start = time.time()
clusters = SpectralClustering(n_clusters=2, affinity="rbf").fit(df)
end = time.time()

# calcula a silhueta dos clusters // calculates the silhouette of the clusters
lable = clusters.labels_
print("silhouette_score:", metrics.silhouette_score(df, lable))

plot = sns.scatterplot(data = df, x = "X", y = "Y", hue = lable, legend = "full", palette = "deep")
sns.move_legend(plot, "upper right", bbox_to_anchor = (1.16, 1), title = "Clusters")

print("execution time:", end - start)
plt.show()